// DFlash2 block drafter (https://inco.ai/blog/dflash2/), ported from
// ml-explore/mlx-swift-lm#607 (spokvulcan) and the reference
// `dflash/model_mlx.py` (https://github.com/z-lab/dflash).
//
// The drafter embeds an `[anchor, MASK, ...]` block, runs it through a few
// bidirectional layers that attend to a window of the target's hidden states,
// keeps the top-K candidates per position, and walks the greedy path through
// them with a pairwise selector. Embedding and LM head are the target's.

import Foundation
import MLX
import MLXNN

// MARK: - Geometry

/// The target shape a drafter was distilled for, or the shape a target has.
public struct DFlash2Geometry: Equatable, Sendable, CustomStringConvertible {
    /// Decoder layers; `nil` when a drafter checkpoint does not record it.
    public var layerCount: Int?
    public var hiddenSize: Int
    public var vocabularySize: Int

    public init(layerCount: Int?, hiddenSize: Int, vocabularySize: Int) {
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
    }

    public var description: String {
        "\(layerCount.map(String.init) ?? "?") layers, hidden \(hiddenSize), vocabulary \(vocabularySize)"
    }

    /// Whether a drafter distilled for `self`, reading `layerIds`, can draft for `target`.
    func accepts(_ target: DFlash2Geometry, layerIds: [Int]) -> Bool {
        guard let targetLayers = target.layerCount else { return false }
        return hiddenSize == target.hiddenSize && vocabularySize == target.vocabularySize
            && (layerCount ?? targetLayers) == targetLayers
            && layerIds.allSatisfy { (0 ..< targetLayers).contains($0) }
    }
}

// MARK: - Drafter protocol

/// One block to propose. Every accept-dependent input may be lazy, so the next
/// proposal can be built while the previous verify pass runs.
public struct DFlash2ProposalRequest {
    /// `[1, width]` token ids: the anchor, then mask tokens.
    public var block: MLXArray
    /// `[1, S, targetLayerIds.count * hidden]` target outputs verified since the last proposal.
    public var targetHidden: MLXArray
    /// Absolute position of `targetHidden` row 0.
    public var contextPosition: Int
    /// `[]` int32 count of leading `targetHidden` rows that are committed context.
    public var validRows: MLXArray
    /// Token ids the target never emits; drafts skip them.
    public var suppressedTokens: SuppressTokensProcessor?

    public init(
        block: MLXArray, targetHidden: MLXArray, contextPosition: Int, validRows: MLXArray,
        suppressedTokens: SuppressTokensProcessor? = nil
    ) {
        self.block = block
        self.targetHidden = targetHidden
        self.contextPosition = contextPosition
        self.validRows = validRows
        self.suppressedTokens = suppressedTokens
    }
}

/// The drafter's greedy path and the per-position candidates it was chosen from.
public struct DFlash2Proposal {
    /// `[1, width - 1]` drafted token ids.
    public var tokens: MLXArray
    /// `[1, width - 1, K]` candidate ids per drafted position.
    public var candidates: MLXArray

    public init(tokens: MLXArray, candidates: MLXArray) {
        self.tokens = tokens
        self.candidates = candidates
    }
}

/// Per-stream drafter state, owned by the iterator so one drafter serves many streams.
public struct DFlash2DrafterState {
    /// One context cache per drafter layer.
    public var contextCaches: [DFlash2ContextCache]

    public init(contextCaches: [DFlash2ContextCache]) {
        self.contextCaches = contextCaches
    }
}

/// A DFlash2 block drafter.
///
/// It attends over projections of the target's hidden states and predicts every
/// position of an `[anchor, MASK, ...]` block at once. It borrows the target's
/// embedding and head per call and holds no per-stream state, like
/// ``MTPDrafterModel``.
public protocol DFlash2DrafterModel: BaseLanguageModel {
    /// Tokens per verify pass the checkpoint was trained for, anchor included.
    var blockSize: Int { get }
    /// Token id that fills the block's non-anchor positions.
    var maskTokenId: Int { get }
    /// Target layers the drafter reads, in feature-concatenation order.
    var targetLayerIds: [Int] { get }
    /// The target shape the drafter was distilled for.
    var targetGeometry: DFlash2Geometry { get }
    /// Target hidden rows the drafter's context keeps.
    var contextWindow: Int { get }

    func makeState() -> DFlash2DrafterState

    /// Propose one block along the drafter's greedy path. `request.targetHidden`
    /// rows enter `state` as placeholders until the iterator resolves them.
    func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal
}

// MARK: - Target protocol

/// One verify pass over `[anchor, draft_1, ..., draft_{S-1}]`.
public struct DFlash2VerifyRequest {
    /// `[1, S]` token ids.
    public var tokens: MLXArray
    /// `[1]` int32 absolute position of `tokens[0]`; may be lazy.
    public var position: MLXArray
    /// Largest value `position` can take. Bounds the attention span.
    public var positionUpperBound: Int
    /// Layers whose outputs the drafter reads, in ``DFlash2VerifyResult/hidden`` order.
    public var captureLayers: [Int]

    public init(
        tokens: MLXArray, position: MLXArray, positionUpperBound: Int, captureLayers: [Int]
    ) {
        self.tokens = tokens
        self.position = position
        self.positionUpperBound = positionUpperBound
        self.captureLayers = captureLayers
    }
}

public struct DFlash2VerifyResult {
    /// `[1, S, vocab]`; row `i` predicts input position `i + 1`.
    public var logits: MLXArray
    /// `[1, S, hidden]` per requested capture layer.
    public var hidden: [MLXArray]
    /// One capture per gated-delta layer, in cache order.
    public var recurrentCaptures: [GatedDeltaCapture]

    public init(logits: MLXArray, hidden: [MLXArray], recurrentCaptures: [GatedDeltaCapture]) {
        self.logits = logits
        self.hidden = hidden
        self.recurrentCaptures = recurrentCaptures
    }
}

/// A hybrid attention / gated-delta target that DFlash2 can speculate for.
///
/// Prefill goes through the ordinary forward: with ``mtpEmitFlagKey`` set and
/// ``mtpTargetLayerIdsKey`` naming layers, the target writes the concatenation
/// of those layers' outputs to ``mtpLastHiddenStatesKey``.
///
/// A verify pass never commits: it writes attention rows at the request's
/// position without moving any offset and returns recurrent state as captures.
/// Its gated-delta scan takes ``gatedDeltaVerifyUpdate(q:k:v:a:b:aLog:dtBias:state:useKernel:)``,
/// so a replayed capture is exactly the state the pass computed. The iterator
/// commits once the accept count is known.
public protocol DFlash2TargetModel: LanguageModel {
    var dflash2Geometry: DFlash2Geometry { get }
    /// Embedding table the drafter borrows.
    var dflash2Embedding: Embedding { get }
    /// LM head the drafter borrows; `nil` when tied to the embedding.
    var dflash2Head: Linear? { get }

    /// Throws ``DFlash2SpeculationError`` when the verify pass cannot drive
    /// `cache` continued from `state`. Called once per generation, before prefill.
    func dflash2Prepare(cache: [KVCache], state: LMOutput.State?) throws

    func dflash2Verify(_ request: DFlash2VerifyRequest, cache: [KVCache]) -> DFlash2VerifyResult
}

// MARK: - Keyed traces

/// Compiled traces built on demand, one per key. `Module.invalidateCompiledTraces()`
/// finds it by reflection when a module stores it, so replacing modules drops every trace.
package final class KeyedCompiledTraces<Key: Hashable, Owner: Module>: CompiledTraceInvalidating,
    @unchecked Sendable
{
    private let lock = NSLock()
    private var traces: [Key: CompiledTrace<Owner>] = [:]

    package init() {}

    package var count: Int { lock.withLock { traces.count } }

    package func trace(_ key: Key, make: () -> CompiledTrace<Owner>) -> CompiledTrace<Owner> {
        lock.withLock {
            if let existing = traces[key] { return existing }
            let trace = make()
            traces[key] = trace
            return trace
        }
    }

    package func invalidate() {
        lock.withLock { traces.removeAll() }
    }
}

// MARK: - Drafter context cache

/// Sliding-window cache of one drafter layer's context keys and values.
///
/// A pipelined round appends the whole verify block before its accept count is
/// known, so rows enter as placeholders with explicit positions and
/// `resolve(newest:valid:)` commits the accepted prefix. Rows are compacted to
/// the newest valid `window` rows once every few hundred appends; until then
/// the attention mask hides placeholders and rows past the window.
public final class DFlash2ContextCache {
    public let window: Int

    private var keyStore: MLXArray?
    private var valueStore: MLXArray?
    private var storedCount = 0
    private let compactionSlack = 256

    /// Absolute position of each stored row.
    package private(set) var rowPositions: [Int32] = []
    /// Whether each stored row is committed context or a placeholder.
    package private(set) var rowValid: [Bool] = []

    public init(window: Int) {
        precondition(window > 0, "DFlash2ContextCache needs a positive window")
        self.window = window
    }

    /// Stored rows, placeholders included.
    public var count: Int { storedCount }

    package var keys: MLXArray? { keyStore?[.ellipsis, ..<storedCount, 0...] }
    package var values: MLXArray? { valueStore?[.ellipsis, ..<storedCount, 0...] }

    /// The backing buffers, so the iterator can evaluate them each round.
    package var innerState: [MLXArray] { [keyStore, valueStore].compactMap { $0 } }

    /// Append `[1, heads, n, dim]` rows as placeholders at `positions`, and return
    /// the stored keys and values.
    @discardableResult
    package func append(
        keys newKeys: MLXArray, values newValues: MLXArray, positions: [Int32]
    ) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(positions.count == n, "one position per appended row")
        if keyStore == nil {
            let capacity = Swift.max(n, window + compactionSlack)
            keyStore = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)], dtype: newKeys.dtype)
            valueStore = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)],
                dtype: newValues.dtype)
        }
        if storedCount + n > keyStore!.dim(2) || storedCount > window + compactionSlack {
            compact(reserving: n)
        }
        keyStore![.ellipsis, storedCount ..< (storedCount + n), 0...] = newKeys
        valueStore![.ellipsis, storedCount ..< (storedCount + n), 0...] = newValues
        storedCount += n
        rowPositions.append(contentsOf: positions)
        rowValid.append(contentsOf: Array(repeating: false, count: n))
        return (keys!, values!)
    }

    /// Commit the first `valid` of the newest `newest` rows. The rest stay
    /// placeholders and drop at the next compaction.
    package func resolve(newest: Int, valid: Int) {
        precondition(newest <= storedCount, "resolving more rows than stored")
        let base = storedCount - newest
        for i in 0 ..< newest {
            rowValid[base + i] = i < valid
        }
    }

    /// Keep the newest `window` valid rows in a fresh padded buffer.
    private func compact(reserving n: Int) {
        let kept = (0 ..< storedCount).filter { rowValid[$0] }.suffix(window)
        let gather = MLXArray(kept.map { Int32($0) })
        let keptKeys = MLX.take(keyStore!, gather, axis: 2)
        let keptValues = MLX.take(valueStore!, gather, axis: 2)
        let capacity = Swift.max(window + compactionSlack, kept.count + n)
        let padK = [keptKeys.dim(0), keptKeys.dim(1), capacity - kept.count, keptKeys.dim(3)]
        let padV = [
            keptValues.dim(0), keptValues.dim(1), capacity - kept.count, keptValues.dim(3),
        ]
        keyStore = concatenated([keptKeys, MLXArray.zeros(padK, dtype: keptKeys.dtype)], axis: 2)
        valueStore = concatenated(
            [keptValues, MLXArray.zeros(padV, dtype: keptValues.dtype)], axis: 2)
        rowPositions = kept.map { rowPositions[$0] }
        rowValid = Array(repeating: true, count: kept.count)
        storedCount = kept.count
    }
}

// MARK: - Configuration

/// The `dflash_config` object of a DFlash2 checkpoint.
public struct DFlash2SpecConfiguration: Codable, Sendable {
    public var blockSize: Int = 16
    public var convGroupSize: Int = 16
    public var convKernelSize: Int = 2
    public var maskTokenId: Int = 0
    public var selectorRank: Int = 256
    public var selectorTopK: Int = 16
    public var targetLayerIds: [Int] = []
    public var inputEmbeddingScale: Float = 1.0
    public var outputMultiplier: Float = 1.0
    public var finalLogitSoftcapping: Float?
    /// The mlx-community layout's sliding window: `use_swa` with `swa_window_size`.
    public var useSlidingWindow: Bool?
    public var slidingWindowSize: Int?

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case convGroupSize = "conv_group_size"
        case convKernelSize = "conv_kernel_size"
        case maskTokenId = "mask_token_id"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case targetLayerIds = "target_layer_ids"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useSlidingWindow = "use_swa"
        case slidingWindowSize = "swa_window_size"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
        convGroupSize = try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 16
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 2
        maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? 0
        selectorRank = try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 256
        selectorTopK = try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 16
        targetLayerIds =
            try container.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
        inputEmbeddingScale =
            try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1.0
        outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1.0
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow)
        slidingWindowSize = try container.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
    }
}

/// Configuration of a DFlash2 draft checkpoint (`model_type: qwen3`,
/// `architectures: [DFlash2DraftModel]`).
///
/// Reads both published layouts: `layer_types` with `sliding_window` (z-lab,
/// inco.ai), and `dflash_config.use_swa` with a top-level `block_size`
/// (mlx-community). Every layer must be non-causal sliding-window attention.
public struct DFlash2Configuration: Decodable, Sendable {
    public var hiddenSize: Int = 5120
    public var hiddenLayers: Int = 5
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 17408
    public var vocabularySize: Int = 248320
    public var rmsNormEps: Float = 1e-6
    public var maxPositionEmbeddings: Int = 262144
    public var slidingWindow: Int = 2048
    public var ropeTheta: Float = 10_000_000
    public var ropeScaling: [String: StringOrNumber]? = nil
    /// Depth of the target the drafter was distilled for; mlx-community omits it.
    public var numTargetLayers: Int?
    public var dflash: DFlash2SpecConfiguration = .init()

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isCausal = "is_causal"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case dflash = "dflash_config"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 5
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / attentionHeads)
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 17408
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248320
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        numTargetLayers = try container.decodeIfPresent(Int.self, forKey: .numTargetLayers)
        dflash =
            try container.decodeIfPresent(DFlash2SpecConfiguration.self, forKey: .dflash)
            ?? .init()
        let nested = try? container.nestedContainer(
            keyedBy: DFlash2SpecConfiguration.CodingKeys.self, forKey: .dflash)
        if try nested?.decodeIfPresent(Int.self, forKey: .blockSize) == nil,
            let blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize)
        {
            dflash.blockSize = blockSize
        }

        var ropeParameters =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if ropeParameters?["type"] == nil, let ropeType = ropeParameters?["rope_type"] {
            ropeParameters?["type"] = ropeType
        }
        let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? topLevelTheta ?? 10_000_000
        ropeScaling = ropeParameters

        let window: Int?
        if let layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            let allSliding =
                layerTypes.count == hiddenLayers
                && layerTypes.allSatisfy { $0 == "sliding_attention" }
            window =
                allSliding ? try container.decodeIfPresent(Int.self, forKey: .slidingWindow) : nil
        } else {
            window = dflash.useSlidingWindow == true ? dflash.slidingWindowSize : nil
        }
        guard let window, window > 1 else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "DFlash2 needs sliding-window attention in every layer")
        }
        slidingWindow = window
        // The reference treats a missing `is_causal` as causal for sliding layers.
        guard try container.decodeIfPresent(Bool.self, forKey: .isCausal) == false else {
            throw DecodingError.dataCorruptedError(
                forKey: .isCausal, in: container,
                debugDescription: "DFlash2 needs a non-causal (is_causal: false) drafter")
        }
        guard !dflash.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids is empty")
        }
    }

    /// Context rows a block can see: the window minus the block's own anchor.
    public var contextWindow: Int { slidingWindow - 1 }
}

// MARK: - Two-tap dynamic depthwise convolution

/// `Conv(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}` on `groupSize`-channel groups:
/// a learned base kernel plus a per-group correction projected from the input.
/// `prepare` applies slot 0 and returns the dynamic kernel; `finish` applies slot 1.
final class DFlash2DynamicConv: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        precondition(hiddenSize % groupSize == 0, "conv groups must divide hidden size")
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// hidden `[B, L, H]`, dynamic `[B, L, K, groups]`, base `[K, H]`. The first
    /// position reads zeros before it.
    private static func convolve(
        _ hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let groups = h / groupSize
        let blocks = hidden.reshaped(b, l, groups, groupSize)
        var output = MLXArray.zeros(like: blocks)
        for tap in 0 ..< base.dim(0) {
            let values: MLXArray
            if tap == 0 {
                values = blocks
            } else {
                values = concatenated(
                    [MLXArray.zeros(like: blocks[0..., ..<tap]), blocks[0..., ..<(l - tap)]],
                    axis: 1)
            }
            let baseKernel = base[tap].asType(hidden.dtype).reshaped(1, 1, groups, groupSize)
            output = output + baseKernel * values
            output = output + dynamic[0..., 0..., tap, 0..., .newAxis] * values
        }
        return output.reshaped(hidden.shape)
    }

    func prepare(_ hidden: MLXArray) -> (convolved: MLXArray, kernel: MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0], groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, kernel: MLXArray) -> MLXArray {
        Self.convolve(hidden, dynamic: kernel, base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - Attention

/// Block queries attend to the cached context K/V plus the block's own K/V.
final class DFlash2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlash2Configuration) {
        numHeads = config.attentionHeads
        numKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    /// Rotated block queries and keys plus values, heads first.
    func projectBlock(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (b, l) = (x.dim(0), x.dim(1))
        let queries = qNorm(qProj(x).reshaped(b, l, numHeads, headDim)).transposed(0, 2, 1, 3)
        let keys = kNorm(kProj(x).reshaped(b, l, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (
            applyRotaryPosition(rope, to: queries, offset: .batch(position)),
            applyRotaryPosition(rope, to: keys, offset: .batch(position)),
            values
        )
    }

    /// Rotated context keys plus values for the cache, heads first.
    func projectContext(
        _ x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        let (b, s) = (x.dim(0), x.dim(1))
        let keys = kNorm(kProj(x).reshaped(b, s, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, s, numKVHeads, headDim).transposed(0, 2, 1, 3)
        return (applyRotaryPosition(rope, to: keys, offset: .batch(position)), values)
    }

    func project(_ attention: MLXArray) -> MLXArray {
        let (b, l) = (attention.dim(0), attention.dim(2))
        return oProj(attention.transposed(0, 2, 1, 3).reshaped(b, l, numHeads * headDim))
    }
}

// MARK: - Decoder layer

/// Pre-norm layer with a dynamic conv pair around each sublayer, split at the
/// attention so the SDPA over the growing context runs between two traces.
final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2DynamicConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2DynamicConv

    init(_ config: DFlash2Configuration) {
        _selfAttn.wrappedValue = DFlash2Attention(config)
        _mlp.wrappedValue = DFlash2MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        _mlpConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        super.init()
    }

    /// Input norm, conv prepare and the block projections.
    func preAttention(
        x: MLXArray, rope: RoPELayer, position: MLXArray
    ) -> (kernel: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (h, kernel) = attentionConv.prepare(inputLayerNorm(x))
        let (queries, keys, values) = selfAttn.projectBlock(h, rope: rope, position: position)
        return (kernel, queries, keys, values)
    }

    /// Output projection, conv finish, both residual adds and the MLP.
    func postAttention(x: MLXArray, attention: MLXArray, kernel: MLXArray) -> MLXArray {
        let h = x + attentionConv.finish(selfAttn.project(attention), kernel: kernel)
        let (m, mlpKernel) = mlpConv.prepare(postAttentionLayerNorm(h))
        return h + mlpConv.finish(mlp(m), kernel: mlpKernel)
    }
}

final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DFlash2Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Candidate selector

/// Scores adjacent pairs of top-K candidates with a low-rank bilinear match
/// gated by the successor's hidden state,
/// `S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>`, then walks the best path.
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    /// The greedy walk is about 60 small ops; one trace per block width.
    private let compiledGreedy = KeyedCompiledTraces<Int, DFlash2CandidateSelector>()

    init(_ config: DFlash2Configuration) {
        topK = config.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - hidden: normed block hidden states past the anchor, `[B, L, H]`.
    ///   - logits: matching draft logits, `[B, L, V]`.
    ///   - anchor: the verified token before the block, `[B]`.
    ///   - compiled: whether to run the walk through its per-width trace.
    func select(
        hidden: MLXArray, logits: MLXArray, anchor: MLXArray, compiled: Bool = true
    ) -> DFlash2Proposal {
        guard compiled else { return walk(hidden: hidden, logits: logits, anchor: anchor) }
        let trace = compiledGreedy.trace(logits.dim(1)) {
            CompiledTrace<DFlash2CandidateSelector> { selector, args in
                let proposal = selector.walk(hidden: args[0], logits: args[1], anchor: args[2])
                return [proposal.tokens, proposal.candidates]
            }
        }
        let outputs = trace(self, [hidden, logits, anchor])
        return DFlash2Proposal(tokens: outputs[0], candidates: outputs[1])
    }

    var compiledTraceCount: Int { compiledGreedy.count }

    private func walk(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> DFlash2Proposal {
        let length = logits.dim(1)
        let kth = logits.dim(-1) - topK
        let candidates = argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let unary = takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)
        let successors = successorCodebook(candidates)
        let predecessors = predecessorCodebook(candidates)
        // Boundary 0 pairs the anchor with position 0; boundary t pairs the
        // candidates of t - 1 with those of t, gated by position t's hidden.
        let anchorEdges =
            ((predecessorCodebook(anchor) * projected[0..., 0])[0..., .newAxis, 0...]
            * successors[0..., 0, 0..., 0...]).sum(axis: -1)
        let gated =
            predecessors[0..., 0 ..< (length - 1), 0..., 0...]
            * projected[0..., 1..., .newAxis, 0...]
        let edges = matmul(gated, successors[0..., 1..., 0..., 0...].transposed(0, 1, 3, 2))

        var path: [MLXArray] = []
        var previous: MLXArray? = nil
        for position in 0 ..< length {
            var scores = unary[0..., position]
            if let previous {
                scores =
                    scores
                    + takeAlong(
                        edges[0..., position - 1, 0..., 0...],
                        previous[0..., .newAxis, .newAxis], axis: 1
                    )[0..., 0, 0...]
            } else {
                scores = scores + anchorEdges
            }
            let chosen = argMax(scores, axis: -1)
            previous = chosen
            path.append(
                takeAlong(candidates[0..., position], chosen[.newAxis, 0...], axis: -1)[0..., 0])
        }
        return DFlash2Proposal(tokens: stacked(path, axis: 1), candidates: candidates)
    }
}

// MARK: - Draft model

/// The DFlash2 drafter.
///
/// A proposal runs as compiled segments split at each layer's attention. The
/// block position and every accept-dependent value enter as arrays, so one
/// graph serves any accept outcome.
public final class DFlash2DraftModel: Module, DFlash2DrafterModel {
    public let config: DFlash2Configuration

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlash2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    let rope: RoPELayer

    /// Test hook: run the segment bodies eagerly instead of through traces.
    var compiledTracesEnabled = true

    public init(_ config: DFlash2Configuration) {
        self.config = config
        let concatDim = config.dflash.targetLayerIds.count * config.hiddenSize
        _fc.wrappedValue = Linear(concatDim, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            DFlash2DecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    // MARK: DFlash2DrafterModel

    public var blockSize: Int { config.dflash.blockSize }
    public var maskTokenId: Int { config.dflash.maskTokenId }
    public var targetLayerIds: [Int] { config.dflash.targetLayerIds }
    public var contextWindow: Int { config.contextWindow }

    public var targetGeometry: DFlash2Geometry {
        DFlash2Geometry(
            layerCount: config.numTargetLayers, hiddenSize: config.hiddenSize,
            vocabularySize: config.vocabularySize)
    }

    public func makeState() -> DFlash2DrafterState {
        DFlash2DrafterState(
            contextCaches: layers.map { _ in DFlash2ContextCache(window: contextWindow) })
    }

    public func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        let block = request.block
        precondition(block.dim(0) == 1, "DFlash2 drafts one stream at a time")
        precondition(
            state.contextCaches.count == layers.count, "one context cache per drafter layer")
        let rows = request.targetHidden.dim(1)
        precondition(rows <= contextWindow, "targetHidden exceeds the drafter's context window")
        let width = block.dim(1)
        let contextPosition = MLXArray([Int32(request.contextPosition)])
        let blockPosition = contextPosition + request.validRows.asType(.int32).reshaped([1])

        // Context K/V for the new rows, appended as placeholders.
        let contextKV = projectContext(request.targetHidden, position: contextPosition)
        let positions = (0 ..< rows).map { Int32(request.contextPosition + $0) }
        var layerKV: [(MLXArray, MLXArray)] = []
        for (i, cache) in state.contextCaches.enumerated() {
            layerKV.append(
                cache.append(
                    keys: contextKV[2 * i], values: contextKV[2 * i + 1], positions: positions))
        }
        // The caches move in lockstep, so one mask serves every layer.
        let mask = Self.visibilityMask(
            cache: state.contextCaches[0], appended: rows, validRows: request.validRows,
            blockPosition: blockPosition, width: width, window: config.slidingWindow)

        // The target's embedding and head run outside the traces.
        var x = target.dflash2Embedding(block) * config.dflash.inputEmbeddingScale
        var attention: MLXArray? = nil
        var kernel: MLXArray? = nil
        for index in 0 ... layers.count {
            let args = attention.map { [x, $0, kernel!, blockPosition] } ?? [x, blockPosition]
            let outputs =
                compiledTracesEnabled
                ? segmentTrace(index: index, width: width)(self, args)
                : segmentBody(at: index, args)
            x = outputs[0]
            guard index < layers.count else { break }
            kernel = outputs[1]
            let (contextKeys, contextValues) = layerKV[index]
            attention = MLXFast.scaledDotProductAttention(
                queries: outputs[2],
                keys: concatenated([contextKeys, outputs[3]], axis: 2),
                values: concatenated([contextValues, outputs[4]], axis: 2),
                scale: layers[index].selfAttn.scale, mask: mask)
        }

        let hidden = x[0..., 1..., 0...]
        var logits = target.dflash2Head?(hidden) ?? target.dflash2Embedding.asLinear(hidden)
        logits = logits * config.dflash.outputMultiplier
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        if let suppressor = request.suppressedTokens {
            logits = suppressor.process(logits: logits[0]).expandedDimensions(axis: 0)
        }
        return candidateSelector.select(
            hidden: hidden, logits: logits, anchor: block[0..., 0],
            compiled: compiledTracesEnabled)
    }

    /// Bool mask `[width, cache.count + width]`. Committed context rows within the
    /// window are visible, placeholders never, and the block sees itself in full.
    static func visibilityMask(
        cache: DFlash2ContextCache, appended: Int, validRows: MLXArray,
        blockPosition: MLXArray, width: Int, window: Int
    ) -> MLXArray {
        let older = cache.count - appended
        let keyPositions = MLXArray(cache.rowPositions)
        let olderValid = MLXArray(Array(cache.rowValid[0 ..< older]))
        let appendedValid = MLXArray(Int32(0) ..< Int32(appended)) .< validRows.asType(.int32)
        let valid = concatenated([olderValid, appendedValid])
        let queryPositions = blockPosition.asType(.int32) + MLXArray(Int32(0) ..< Int32(width))
        let distance =
            queryPositions.expandedDimensions(axis: 1) - keyPositions.expandedDimensions(axis: 0)
        let contextVisible = valid.expandedDimensions(axis: 0) & (distance .< Int32(window))
        return concatenated(
            [contextVisible, MLXArray.ones([width, width], dtype: .bool)], axis: 1)
    }

    // MARK: Compiled segments

    private struct SegmentKey: Hashable {
        var index: Int
        var width: Int
    }

    private let segmentTraces = KeyedCompiledTraces<SegmentKey, DFlash2DraftModel>()
    private let contextTraces = KeyedCompiledTraces<Int, DFlash2DraftModel>()

    /// The prompt window projects eagerly; a trace per prompt length would only churn.
    private static let maxTracedContextRows = 16

    /// Live traces, so tests can check that the compiled route ran.
    var compiledTraceCount: Int {
        segmentTraces.count + contextTraces.count + candidateSelector.compiledTraceCount
    }

    /// Segment `i` closes layer `i - 1` and opens layer `i`; the last segment
    /// applies the final norm. In: `[x, position]` for segment 0, else
    /// `[x, attention, kernel, position]`. Out: `[x, kernel, queries, keys, values]`,
    /// or `[normed]` for the last.
    private func segmentBody(at index: Int, _ args: [MLXArray]) -> [MLXArray] {
        var x = args[0]
        if index > 0 {
            x = layers[index - 1].postAttention(x: x, attention: args[1], kernel: args[2])
        }
        guard index < layers.count else { return [norm(x)] }
        let position = args[index == 0 ? 1 : 3]
        let (kernel, queries, keys, values) = layers[index].preAttention(
            x: x, rope: rope, position: position)
        return [x, kernel, queries, keys, values]
    }

    private func segmentTrace(index: Int, width: Int) -> CompiledTrace<DFlash2DraftModel> {
        segmentTraces.trace(SegmentKey(index: index, width: width)) {
            CompiledTrace<DFlash2DraftModel> { model, args in model.segmentBody(at: index, args) }
        }
    }

    /// `[targetHidden, position]` to `[keys, values]` per layer.
    private func contextBody(_ args: [MLXArray]) -> [MLXArray] {
        let context = hiddenNorm(fc(args[0]))
        return layers.flatMap { layer in
            let (keys, values) = layer.selfAttn.projectContext(
                context, rope: rope, position: args[1])
            return [keys, values]
        }
    }

    private func projectContext(_ targetHidden: MLXArray, position: MLXArray) -> [MLXArray] {
        let rows = targetHidden.dim(1)
        guard compiledTracesEnabled, rows <= Self.maxTracedContextRows else {
            return contextBody([targetHidden, position])
        }
        let trace = contextTraces.trace(rows) {
            CompiledTrace<DFlash2DraftModel> { model, args in model.contextBody(args) }
        }
        return trace(self, [targetHidden, position])
    }

    /// The bf16 checkpoint stores the selector codebooks as bare tensors;
    /// `Embedding` expects a `.weight` suffix. A quantized artifact already
    /// writes `.weight`, `.scales` and `.biases`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let bare = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: bare) {
                weights["\(bare).weight"] = value
            }
        }
        return weights
    }
}

// MARK: - Loading

/// A loaded DFlash2 drafter. It has no tokenizer or chat template of its own.
/// Not `Sendable`; share it through ``DFlash2DrafterContainer``.
public struct DFlash2DrafterContext {
    public var configuration: ModelConfiguration
    public var model: any DFlash2DrafterModel

    public init(configuration: ModelConfiguration, model: any DFlash2DrafterModel) {
        self.configuration = configuration
        self.model = model
    }
}

/// Sendable container for a ``DFlash2DrafterContext``, in the shape of
/// ``ModelContainer``.
public final class DFlash2DrafterContainer: ModelMemoryReporting, Sendable {
    private let context: SerialAccessContainer<DFlash2DrafterContext>

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public init(context: consuming DFlash2DrafterContext) {
        self.context = .init(context)
    }

    /// Perform an action on the context. Evaluate any `MLXArray` before returning it.
    public func perform<R: Sendable>(
        _ action: @Sendable (DFlash2DrafterContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    public func memory() async throws -> ModelMemory {
        try await context.read { try $0.model.memory() }
    }
}

/// Loads a DFlash2 drafter from a local checkpoint directory.
///
/// A quantized artifact is loaded as stored: the loader quantizes exactly the
/// layers whose `.scales` the checkpoint carries, with the `quantization`
/// block of its `config.json`.
public enum DFlash2DrafterModelFactory {
    public static func load(from directory: URL) async throws -> DFlash2DrafterContext {
        let configurationURL = directory.appending(component: "config.json")
        let name = directory.lastPathComponent
        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, name, error)
        }
        let baseConfig: BaseConfiguration
        let config: DFlash2Configuration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
            config = try JSONDecoder.json5().decode(DFlash2Configuration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, name, error)
        }
        let model = DFlash2DraftModel(config)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)
        return DFlash2DrafterContext(
            configuration: ModelConfiguration(directory: directory, defaultPrompt: ""),
            model: model)
    }

    public static func loadContainer(from directory: URL) async throws -> DFlash2DrafterContainer {
        DFlash2DrafterContainer(context: try await load(from: directory))
    }
}
