// DFlash2 speculative token iterator. The round pipeline, recurrent commit and
// rewind are ported from ml-explore/mlx-swift-lm#607 (spokvulcan); acceptance is
// sample-and-match rather than the reference's rejection sampling.

import Foundation
import MLX

/// Why a ``DFlash2TokenIterator`` could not be built.
public enum DFlash2SpeculationError: Error, Equatable, LocalizedError {
    /// The target model does not conform to ``DFlash2TargetModel``.
    case unsupportedTarget
    /// A block holds the anchor and at least one draft.
    case invalidBlockSize(Int)
    /// The drafter was distilled for a target of another shape.
    case geometryMismatch(drafter: DFlash2Geometry, target: DFlash2Geometry)
    /// DFlash2 speculation covers text-only prompts.
    case textOnly
    /// The iterator decodes one sequence.
    case batchUnsupported(Int)
    /// `prefilledPrefixTokens` must leave at least one prompt token.
    case promptTooShort(promptTokens: Int, prefilledPrefixTokens: Int)
    /// The cache does not hold exactly `prefilledPrefixTokens` positions.
    case cacheOffsetMismatch(cacheOffset: Int, prefilledPrefixTokens: Int)
    /// The cache or the requested KV cache configuration has entries the verify
    /// pass cannot write, such as quantized or rotating attention caches.
    case unsupportedCache(String)
    /// A warm cache came without the model state its positions continue from.
    case missingContinuationState(String)
    /// The target's positions are not plain offsets, such as a nonzero M-RoPE delta.
    case unsupportedPositions(String)
    /// A round that costs `roundCost` plain steps yields at most `blockSize`
    /// tokens, so at or above it no acceptance pays.
    case roundCostCannotPay(roundCost: Double, blockSize: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedTarget:
            "DFlash2 speculation needs a target that conforms to DFlash2TargetModel."
        case .invalidBlockSize(let size):
            "DFlash2 block size must be at least 2; got \(size)."
        case .geometryMismatch(let drafter, let target):
            "The DFlash2 drafter was distilled for \(drafter); the target has \(target)."
        case .textOnly:
            "DFlash2 speculation covers text-only prompts."
        case .batchUnsupported(let batch):
            "DFlash2 speculation decodes one sequence; got a batch of \(batch)."
        case .promptTooShort(let prompt, let prefix):
            "prefilledPrefixTokens (\(prefix)) must be less than the prompt length (\(prompt))."
        case .cacheOffsetMismatch(let offset, let prefix):
            "The cache holds \(offset) positions; prefilledPrefixTokens is \(prefix)."
        case .unsupportedCache(let reason):
            "DFlash2 cannot verify over this cache: \(reason)."
        case .missingContinuationState(let reason):
            "DFlash2 cannot continue this cache: \(reason)."
        case .unsupportedPositions(let reason):
            "DFlash2 cannot position this prompt: \(reason)."
        case .roundCostCannotPay(let cost, let block):
            "A DFlash2 round costs \(cost) plain steps, at or above its block of \(block)."
        }
    }
}

/// When a turn stops speculating and finishes with plain single-token steps.
///
/// A verify round costs ``roundCost`` plain steps and yields `τ` tokens, so it
/// loses `roundCost − τ` plain steps against plain decoding. The iterator keeps
/// the running loss `S = max(0, S + roundCost − τ)` over its synced rounds and
/// stops speculating, for the rest of the turn, once `S` reaches
/// ``lossBudget``; a winning stretch pays back an earlier loss.
public struct DFlash2Fallback: Sendable, Equatable {
    /// Seconds per verify round over seconds per plain step on this host, at
    /// the block width the turn uses.
    public var roundCost: Double
    /// Plain steps of running loss at which the turn stops speculating.
    ///
    /// A turn loses at most `lossBudget + 2 (roundCost − 1)` plain steps: the
    /// round that crosses the budget can overshoot it by up to `roundCost − 1`,
    /// and the round already in flight, which still commits, adds up to
    /// `roundCost − 1` more.
    public var lossBudget: Double

    public init(roundCost: Double, lossBudget: Double) {
        self.roundCost = roundCost
        self.lossBudget = lossBudget
    }

    /// The fallback at `blockSize`, with that width's ``lossBudget(forBlockSize:)``.
    public init(roundCost: Double, blockSize: Int) {
        self.init(roundCost: roundCost, lossBudget: Self.lossBudget(forBlockSize: blockSize))
    }

    /// The smallest budget that keeps 97% of the ungated gain on turns at
    /// `τ/R > 1.15`, replayed on recorded acceptance (13 prompts, T 0.15) against
    /// round costs 1–6 with `Benchmarks/DFlash2Bench/replay_budget.py`.
    ///
    /// Wider blocks vary more per round, so they need more room. Replayed:
    /// 5 → 16, 6 → 24, 8 → 40. The rule is `16 + 8 (blockSize − 5)`, so 7 → 32
    /// is interpolated; below 5 it stays 16, and above 8 it is extrapolated and
    /// unreplayed.
    public static func lossBudget(forBlockSize blockSize: Int) -> Double {
        Double(16 + 8 * Swift.max(0, blockSize - 5))
    }

    /// Never stops speculating, for benches and tests.
    public static let disabled = DFlash2Fallback(roundCost: 0, lossBudget: .infinity)
}

/// Token iterator for DFlash2 block speculative decoding.
///
/// Each round verifies `[anchor, draft_1, ..., draft_{S-1}]` in one target pass,
/// where the drafts are the drafter's greedy path. Every verify row is processed
/// and sampled with the generation's own logit processor and sampler, and drafts
/// are accepted while they equal those samples; the first sample that differs
/// is the round's last token. Every emitted token is therefore the target's own
/// sample, at any temperature.
///
/// Sampling is seeded per absolute position, so at equal numerics the output
/// equals plain decoding under the same seeding token for token. A verify pass
/// sums in a different order than a one-token step, so it can differ where the
/// top logits nearly tie.
///
/// The next round is built while the current verify pass runs: its accept
/// count, last token and position stay lazy, so the host syncs once per round.
public struct DFlash2TokenIterator: TokenIteratorProtocol {
    let target: any DFlash2TargetModel
    let drafter: any DFlash2DrafterModel

    /// The target cache, positioned at ``anchorPosition`` between rounds.
    public let cache: [KVCache]
    private var targetState: LMOutput.State?
    private var drafterState: DFlash2DrafterState

    private var processor: LogitProcessor?
    private let parameters: GenerateParameters
    private let baseSeed: UInt64
    /// Masks suppressed ids in the draft logits; the processor masks them in the target's.
    private let draftSuppressor: SuppressTokensProcessor?

    public let maxTokens: Int?
    /// Tokens per verify pass: one anchor plus `blockSize - 1` drafts.
    public let blockSize: Int
    public let fallback: DFlash2Fallback

    public private(set) var promptPrefillTime: TimeInterval = 0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Drafts the last round accepted, for ``finalizeGeneration()``.
    private var committedAcceptedCount = 0
    /// Tokens produced so far, drained or pending.
    private var producedTokens = 0
    /// The newest produced token, which the cache does not hold yet.
    private var lastToken = 0

    /// A round whose verify pass is scheduled but whose accept is unknown.
    private struct Round {
        var drafts: MLXArray
        var width: Int
        var verify: DFlash2VerifyResult
    }

    private var inFlight: Round?
    /// Absolute position of the next input token: the cache's committed length.
    public private(set) var anchorPosition = 0
    /// Recurrent captures of the last committed round, for ``finalizeGeneration()``.
    private var committedCaptures: [GatedDeltaCapture] = []

    /// Plain decoding: the token sampled after feeding ``lastToken``, scheduled
    /// but not yet read.
    private var plainStep: MLXArray?
    private let plainStepsVerify: Bool
    private var passthrough = false
    /// Why the turn stopped speculating, or nil while it speculates.
    public private(set) var passthroughReason: String?
    /// The running loss `S` of ``DFlash2Fallback``, in plain steps.
    public private(set) var speculationLoss: Double = 0

    /// Times each phase of a round behind an evaluation barrier, for the bench.
    package var phaseTimer: DFlash2PhaseTimer?

    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }
    public private(set) var acceptedCount = 0
    public private(set) var proposedCount = 0

    /// - Parameters:
    ///   - input: the whole prompt, text only, one sequence.
    ///   - mainModel: the target; must conform to ``DFlash2TargetModel``.
    ///   - drafter: a DFlash2 drafter distilled for `mainModel`'s shape.
    ///   - mainCache: a target cache to decode over; a new one when nil.
    ///   - prefilledPrefixTokens: leading positions of `input` that `mainCache`
    ///     already holds; only the rest is prefilled.
    ///   - state: the target's state for `mainCache`, such as the M-RoPE delta a
    ///     warm Qwen 3.5 cache continues from.
    ///   - parameters: sampling, prefill and generation limits.
    ///   - blockSize: tokens per verify pass.
    ///   - fallback: when the turn stops speculating; carries this host's
    ///     measured round cost at `blockSize`.
    ///   - components: generation hooks such as a custom logit processor.
    /// - Throws: ``DFlash2SpeculationError`` before any prefill, including
    ///   ``DFlash2SpeculationError/roundCostCannotPay(roundCost:blockSize:)``.
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        mainCache: [KVCache]? = nil,
        prefilledPrefixTokens: Int = 0,
        state: LMOutput.State? = nil,
        parameters: GenerateParameters,
        blockSize: Int = 5,
        fallback: DFlash2Fallback,
        components: GenerationComponents = .init()
    ) throws {
        try self.init(
            input: input, mainModel: mainModel, drafter: drafter, mainCache: mainCache,
            prefilledPrefixTokens: prefilledPrefixTokens, state: state, parameters: parameters,
            blockSize: blockSize, fallback: fallback, components: components, speculates: true)
    }

    /// `speculates: false` decodes with plain steps under the same prefill and
    /// per-position seeding, for equivalence tests and benchmarks.
    /// `plainStepsVerify` runs each plain step as a one-row verify pass, so it
    /// takes the verify pass's gated-delta and attention routes.
    package init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        mainCache: [KVCache]? = nil,
        prefilledPrefixTokens: Int = 0,
        state: LMOutput.State? = nil,
        parameters: GenerateParameters,
        blockSize: Int = 5,
        fallback: DFlash2Fallback = .disabled,
        components: GenerationComponents = .init(),
        speculates: Bool,
        plainStepsVerify: Bool = false
    ) throws {
        guard let target = mainModel as? any DFlash2TargetModel else {
            throw DFlash2SpeculationError.unsupportedTarget
        }
        guard blockSize >= 2 else { throw DFlash2SpeculationError.invalidBlockSize(blockSize) }
        guard fallback.roundCost < Double(blockSize) else {
            throw DFlash2SpeculationError.roundCostCannotPay(
                roundCost: fallback.roundCost, blockSize: blockSize)
        }
        guard
            drafter.targetGeometry.accepts(target.dflash2Geometry, layerIds: drafter.targetLayerIds)
        else {
            throw DFlash2SpeculationError.geometryMismatch(
                drafter: drafter.targetGeometry, target: target.dflash2Geometry)
        }
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw DFlash2SpeculationError.textOnly
        }
        let tokens = input.text.tokens
        if tokens.ndim > 1, tokens.dim(0) != 1 {
            throw DFlash2SpeculationError.batchUnsupported(tokens.dim(0))
        }
        let promptLength = tokens.dim(-1)
        guard prefilledPrefixTokens >= 0, prefilledPrefixTokens < promptLength else {
            throw DFlash2SpeculationError.promptTooShort(
                promptTokens: promptLength, prefilledPrefixTokens: prefilledPrefixTokens)
        }
        // The verify pass writes plain attention rows: no compression or capacity.
        if let configuration = try parameters.kvCachePlan().configuration,
            configuration.capacity != nil || configuration.strategy != .fullPrecision
        {
            throw DFlash2SpeculationError.unsupportedCache(
                "the KV cache plan compresses or bounds the cache")
        }
        let cache = try mainCache ?? target.newCache(parameters: parameters)
        try target.dflash2Prepare(cache: cache, state: state)
        // A gated-delta cache does not count positions; the attention caches do.
        for entry in cache where !(entry is MambaCache) && entry.offset != prefilledPrefixTokens {
            throw DFlash2SpeculationError.cacheOffsetMismatch(
                cacheOffset: entry.offset, prefilledPrefixTokens: prefilledPrefixTokens)
        }
        try components.validate(parameters: parameters)

        self.target = target
        self.drafter = drafter
        self.cache = cache
        self.targetState = state
        self.drafterState = drafter.makeState()
        self.processor = components.logitProcessor(parameters: parameters, model: mainModel)
        self.draftSuppressor = makeSuppressTokensProcessor(model: mainModel)
        self.parameters = parameters
        self.baseSeed = parameters.seed ?? UInt64.random(in: .min ... .max)
        self.maxTokens = parameters.maxTokens
        self.blockSize = blockSize
        self.fallback = fallback
        self.passthrough = !speculates
        self.plainStepsVerify = plainStepsVerify
        if !speculates {
            passthroughReason = "speculation disabled"
        }

        let prefillStart = Date.timeIntervalSinceReferenceDate
        try prefill(
            promptTokens: tokens.ndim > 1 ? tokens[0] : tokens,
            prefilledPrefixTokens: prefilledPrefixTokens, prefill: parameters.prefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    // MARK: Sampling

    /// The generation's sampler, seeded for the token at `position`.
    private func sampler(at position: Int) -> LogitSampler {
        var parameters = parameters
        parameters.seed = Self.seed(base: baseSeed, position: position)
        return parameters.sampler()
    }

    /// SplitMix64 of the base seed advanced by `position` golden-ratio steps.
    static func seed(base: UInt64, position: Int) -> UInt64 {
        var z = base &+ UInt64(truncatingIfNeeded: position) &* 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Sample `[1, vocab]` logits as the token at `position`, through the processor.
    private mutating func sample(_ logits: MLXArray, position: Int) -> MLXArray {
        let processed = processor?.process(logits: logits) ?? logits
        let token = sampler(at: position).sample(logits: processed)
        processor?.didSample(token: token)
        return token
    }

    /// `[S]` samples of `[S, vocab]` verify logits; row `i` is the token at
    /// `firstPosition + i`. Row `i` goes through a processor copy that has seen
    /// drafts `0 ..< i`: it is used only when those drafts were accepted, so that
    /// is its history.
    private func sampleRows(_ logits: MLXArray, drafts: MLXArray, firstPosition: Int) -> MLXArray {
        var scratch = processor?.copy()
        var samples: [MLXArray] = []
        for i in 0 ..< logits.dim(0) {
            var row = logits[i ..< (i + 1)]
            row = scratch?.process(logits: row) ?? row
            samples.append(
                sampler(at: firstPosition + i).sample(logits: row).asType(.int32).reshaped([1]))
            if i < drafts.dim(0) {
                scratch?.didSample(token: drafts[i ..< (i + 1)])
            }
        }
        return concatenated(samples)
    }

    // MARK: Prefill

    /// Chunked prefill that captures the drafter's target layers and keeps the
    /// last `contextWindow` rows as its first context, then samples the first
    /// token and builds round 0.
    private mutating func prefill(
        promptTokens: MLXArray, prefilledPrefixTokens: Int, prefill: PrefillParameters
    ) throws {
        processor?.prompt(promptTokens)
        prefill.prefix?(prefilledPrefixTokens)
        let promptLength = promptTokens.dim(0)

        let window = drafter.contextWindow
        var state = targetState ?? LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerIdsKey] = drafter.targetLayerIds
        var hiddenWindow: MLXArray? = nil
        var hiddenOffset = prefilledPrefixTokens
        func forward(_ range: Range<Int>) -> MLXArray {
            let chunk = LMInput.Text(tokens: promptTokens[range].expandedDimensions(axis: 0))
            let output = target(chunk, cache: cache, state: state)
            guard let outputState = output.state, var hidden = outputState[mtpLastHiddenStatesKey]
            else {
                preconditionFailure("a DFlash2 target emits the drafter's layers when asked")
            }
            state = outputState
            if let existing = hiddenWindow {
                hidden = concatenated([existing, hidden], axis: 1)
            }
            let rows = hidden.dim(1)
            if rows > window {
                hiddenOffset += rows - window
                hidden = hidden[0..., (rows - window)..., 0...]
            }
            hiddenWindow = hidden
            return output.logits
        }

        let total = promptLength - prefilledPrefixTokens
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(
                (range.lowerBound + prefilledPrefixTokens)
                    ..< (range.upperBound
                        + prefilledPrefixTokens))
            asyncEval(cache.flatMap { $0.innerState() } + [hiddenWindow!])
        }
        let logits = forward((prefilledPrefixTokens + processed) ..< promptLength)
        prefill.progress?(total, total)
        targetState = Self.persistentState(state)

        let token = sample(logits[0..., -1, 0...], position: promptLength)
        eval(token)
        lastToken = token.item(Int.self)
        pendingTokens.append(lastToken)
        producedTokens = 1
        anchorPosition = promptLength
        guard !passthrough else { return }

        // The prompt rows are committed context as soon as round 0 appends them.
        let context = hiddenWindow!
        let contextRows = buildRound(
            anchor: token.reshaped([1]).asType(.int32),
            validRows: MLXArray(Int32(context.dim(1))),
            targetHidden: context,
            contextPosition: hiddenOffset,
            position: MLXArray([Int32(promptLength)]),
            positionUpperBound: promptLength,
            producedAtLeast: 1)
        for contextCache in drafterState.contextCaches {
            contextCache.resolve(newest: contextRows, valid: contextRows)
        }
    }

    /// `state` without the drafter exchange keys, so a later plain call does not emit them.
    static func persistentState(_ state: LMOutput.State) -> LMOutput.State? {
        var state = state
        state[mtpEmitFlagKey] = nil
        state[mtpTargetLayerIdsKey] = nil
        state[mtpLastHiddenStatesKey] = nil
        state[mtpSharedKVStatesKey] = nil
        state[mtpSharedKVOffsetsKey] = nil
        state[mtpSharedKVSourceIndicesKey] = nil
        state[mtpPositionDeltasKey] = nil
        return state.isEmpty ? nil : state
    }

    // MARK: Rounds

    /// Drafts in the next round, fewer near `maxTokens`. `producedAtLeast` is
    /// the fewest tokens produced before that round runs.
    private func draftCount(producedAtLeast: Int) -> Int? {
        guard let maxTokens else { return blockSize - 1 }
        let remaining = maxTokens - producedAtLeast
        return remaining > 0 ? Swift.min(blockSize - 1, remaining) : nil
    }

    /// Schedule one round from possibly lazy inputs: the proposal, then the
    /// verify pass over it. Returns the context rows appended to the drafter.
    @discardableResult
    private mutating func buildRound(
        anchor: MLXArray, validRows: MLXArray, targetHidden: MLXArray,
        contextPosition: Int, position: MLXArray, positionUpperBound: Int,
        producedAtLeast: Int
    ) -> Int {
        guard let drafts = draftCount(producedAtLeast: producedAtLeast) else {
            inFlight = nil
            return 0
        }
        let masks = MLXArray(Array(repeating: Int32(drafter.maskTokenId), count: drafts))
        let block = concatenated([anchor, masks]).expandedDimensions(axis: 0)
        let proposal = drafter.propose(
            DFlash2ProposalRequest(
                block: block, targetHidden: targetHidden, contextPosition: contextPosition,
                validRows: validRows, suppressedTokens: draftSuppressor),
            target: target, state: &drafterState)
        let draftTokens = proposal.tokens.flattened().asType(.int32)
        phaseTimer?.mark(.draft, [draftTokens])
        let request = DFlash2VerifyRequest(
            tokens: concatenated([anchor, draftTokens]).expandedDimensions(axis: 0),
            position: position,
            positionUpperBound: positionUpperBound,
            captureLayers: drafter.targetLayerIds)
        let verify = target.dflash2Verify(request, cache: cache)
        phaseTimer?.mark(
            .verify, [verify.logits] + verify.hidden + verify.recurrentCaptures.flatMap(\.arrays))
        asyncEval(verify.logits)
        inFlight = Round(drafts: draftTokens, width: drafts + 1, verify: verify)
        return targetHidden.dim(1)
    }

    /// Whether the rounds so far have lost the budget.
    private var fallbackDue: Bool { speculationLoss >= fallback.lossBudget }

    /// Accept the in-flight round, build the next one while its verify pass
    /// still runs, then sync once and commit.
    private mutating func speculateRound() {
        guard let round = inFlight else { return }
        inFlight = nil
        let gamma = round.width - 1
        phaseTimer?.start()

        // Accept, lazily: drafts up to the first verify sample that differs.
        let samples = sampleRows(
            round.verify.logits[0], drafts: round.drafts, firstPosition: anchorPosition + 1)
        let matches = (round.drafts .== samples[0 ..< gamma]).asType(.int32)
        let accepted = cumprod(matches).sum()
        let bonus = takeAlong(samples, accepted.reshaped([1]), axis: 0)
        let packed = concatenated([round.drafts, accepted.reshaped([1]).asType(.int32), bonus])
        phaseTimer?.mark(.sampling, [packed])
        asyncEval(packed)

        let validCount = accepted + 1
        Self.commitRecurrentState(
            cache: cache, captures: round.verify.recurrentCaptures, validCount: validCount)
        phaseTimer?.mark(
            .replay, cache.compactMap { $0 as? MambaCache }.flatMap { $0.innerState() })

        // Decided on the rounds already synced, so no verify pass is wasted.
        let stopsSpeculating = fallbackDue
        let appendedRows =
            stopsSpeculating
            ? 0
            : buildRound(
                anchor: bonus,
                validRows: validCount,
                targetHidden: concatenatedHidden(round.verify.hidden),
                contextPosition: anchorPosition,
                position: MLXArray([Int32(anchorPosition)]) + validCount.asType(.int32),
                positionUpperBound: anchorPosition + round.width,
                producedAtLeast: producedTokens + 1)
        asyncEval(
            cache.flatMap { $0.innerState() }
                + drafterState.contextCaches.flatMap(\.innerState))

        // The round's one host sync.
        let values = packed.asArray(Int32.self)
        let drafts = values[0 ..< gamma].map(Int.init)
        let acceptedDrafts = Int(values[gamma])
        let last = Int(values[gamma + 1])

        let committed = anchorPosition + acceptedDrafts + 1
        for case let attention as KVCacheSimple in cache {
            attention.commitRows(count: committed)
        }
        if appendedRows > 0 {
            for contextCache in drafterState.contextCaches {
                contextCache.resolve(newest: appendedRows, valid: acceptedDrafts + 1)
            }
        }
        for token in drafts.prefix(acceptedDrafts) {
            processor?.didSample(token: MLXArray([Int32(token)]))
        }
        processor?.didSample(token: MLXArray([Int32(last)]))

        var newTokens = Array(drafts.prefix(acceptedDrafts))
        newTokens.append(last)
        if let maxTokens {
            newTokens = Array(newTokens.prefix(Swift.max(0, maxTokens - tokenCount)))
        }
        pendingTokens.append(contentsOf: newTokens)
        committedAcceptedCount = acceptedDrafts
        producedTokens += acceptedDrafts + 1
        lastToken = last
        anchorPosition = committed
        committedCaptures = round.verify.recurrentCaptures

        proposedCount += gamma
        acceptedCount += acceptedDrafts
        telemetry.recordRound(
            drafted: gamma, accepted: acceptedDrafts, targetVerified: gamma + 1,
            draftModelCalls: 1)
        speculationLoss = Swift.max(
            0, speculationLoss + fallback.roundCost - Double(acceptedDrafts + 1))
        phaseTimer?.mark(
            .commit,
            cache.flatMap { $0.innerState() } + drafterState.contextCaches.flatMap(\.innerState))

        if stopsSpeculating {
            stopSpeculating(
                reason: String(
                    format:
                        "speculation lost %.2f plain steps, budget %.2f at round cost %.2f, after %d rounds",
                    speculationLoss, fallback.lossBudget, fallback.roundCost,
                    telemetry.roundCount))
        }
    }

    private func concatenatedHidden(_ layers: [MLXArray]) -> MLXArray {
        layers.count == 1 ? layers[0] : concatenated(layers, axis: -1)
    }

    /// Give every gated-delta layer the state it would hold after the first
    /// `validCount` positions of the pass. Exact for any count, so it is built
    /// before the count is known.
    static func commitRecurrentState(
        cache: [KVCache], captures: [GatedDeltaCapture], validCount: MLXArray
    ) {
        let recurrentCaches = cache.compactMap { $0 as? MambaCache }
        precondition(
            recurrentCaches.count == captures.count,
            "\(captures.count) recurrent captures for \(recurrentCaches.count) recurrent caches")
        for (recurrent, capture) in zip(recurrentCaches, captures) {
            let replayed = capture.replay(validCount: validCount)
            recurrent[0] = replayed.conv
            recurrent[1] = replayed.recurrent
        }
    }

    // MARK: Plain decoding

    /// Finish the turn with plain steps on the same cache and state.
    private mutating func stopSpeculating(reason: String) {
        passthrough = true
        passthroughReason = reason
        inFlight = nil
        drafterState = DFlash2DrafterState(contextCaches: [])
    }

    /// Feed ``lastToken`` and schedule the token after it.
    private mutating func schedulePlainStep() {
        let input = MLXArray([Int32(lastToken)]).reshaped(1, 1)
        let logits: MLXArray
        if plainStepsVerify {
            let verify = target.dflash2Verify(
                DFlash2VerifyRequest(
                    tokens: input, position: MLXArray([Int32(anchorPosition)]),
                    positionUpperBound: anchorPosition, captureLayers: []),
                cache: cache)
            Self.commitRecurrentState(
                cache: cache, captures: verify.recurrentCaptures, validCount: MLXArray(Int32(1)))
            for case let attention as KVCacheSimple in cache {
                attention.commitRows(count: anchorPosition + 1)
            }
            logits = verify.logits
        } else {
            let output = target(LMInput.Text(tokens: input), cache: cache, state: targetState)
            targetState = output.state
            logits = output.logits
        }
        anchorPosition += 1
        let token = sample(logits[0..., -1, 0...], position: anchorPosition)
        asyncEval([token] + cache.flatMap { $0.innerState() })
        plainStep = token
        producedTokens += 1
    }

    /// The next plain token. The step after it is scheduled before returning,
    /// as ``TokenIterator`` does, so the cache then holds the returned token.
    private mutating func nextPlainToken() -> Int? {
        if plainStep == nil {
            guard maxTokens.map({ producedTokens < $0 }) ?? true else { return nil }
            schedulePlainStep()
        }
        let token = plainStep!.item(Int.self)
        plainStep = nil
        lastToken = token
        if maxTokens.map({ producedTokens < $0 }) ?? true {
            schedulePlainStep()
        }
        return token
    }

    // MARK: TokenIteratorProtocol

    /// The target's state for ``cache``, without the drafter exchange keys.
    public var state: LMOutput.State? { targetState }

    public var tokenCount: Int { telemetry.emittedTokenCount }

    public var reasoningTokenCount: Int? {
        (processor as? any GenerationReasoningTokenCounting)?.generationReasoningTokenCount
    }

    public mutating func recordEmittedToken(_ token: Int) {
        processor?.didEmit(token: token)
    }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        if pendingIndex == pendingTokens.count {
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            committedAcceptedCount = 0
            if passthrough {
                guard let token = autoreleasepool(invoking: { nextPlainToken() }) else {
                    return nil
                }
                pendingTokens.append(token)
            } else {
                autoreleasepool { speculateRound() }
                if pendingTokens.isEmpty {
                    return nil
                }
            }
        }
        // Return freed buffers after prefill and every 256 tokens, as TokenIterator does.
        if tokenCount % 256 == 0 {
            Memory.clearCache()
        }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}

extension DFlash2TokenIterator: GenerationFinalizingTokenIterator {
    /// Rewind the cache to the tokens the consumer drained. Pending draft `j`
    /// sits at verify position `j + 1`; a round's last token has no cache row
    /// yet. After this the cache holds the prompt and every drained token,
    /// except a drained round-final token, which it never held.
    public mutating func finalizeGeneration() {
        defer { processor?.finalizeGeneration() }
        let kept = Swift.min(pendingIndex, committedAcceptedCount)
        let rewind = committedAcceptedCount - kept
        guard rewind > 0 else { return }
        for entry in cache where !(entry is MambaCache) {
            entry.trim(rewind)
        }
        Self.commitRecurrentState(
            cache: cache, captures: committedCaptures, validCount: MLXArray(Int32(kept + 1)))
        eval(cache.flatMap { $0.innerState() })
        anchorPosition -= rewind
        committedAcceptedCount = kept
    }
}

extension DFlash2TokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
}
