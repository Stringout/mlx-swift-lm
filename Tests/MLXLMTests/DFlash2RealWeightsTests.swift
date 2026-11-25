// Real-weights DFlash2 correctness checks, opt-in: set DFLASH2_TARGET_DIR (an
// MLX Qwen3.8 VLM checkpoint), DFLASH2_DRAFTER_DIR (a DFlash2 drafter) and
// DFLASH2_TOKENS_DIR (prerendered `{label, phase, tokenIds}` prompts). Through
// xcodebuild, prefix each with TEST_RUNNER_. DFLASH2_TOKEN_FILES names the
// prompts to run, comma-separated and without `.json`; DFLASH2_TF_WIDTHS the
// teacher-forced widths (default `5,8`); DFLASH2_BLOCK the width of the
// calibration and of the three speculative-decoding tests (default 5). Each test
// prints its measurements.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

private struct NoTokenizerLoader: TokenizerLoader {
    struct NoTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    func load(from directory: URL) async throws -> any Tokenizer { NoTokenizer() }
}

private struct PromptFile: Decodable {
    let label: String
    let phase: String
    let tokenIds: [Int]
}

/// The loaded pair, shared by every test in the process.
private enum RealWeights {
    nonisolated(unsafe) static var loaded: (target: Qwen35, drafter: DFlash2DraftModel)?

    static func environment(_ key: String) -> String? {
        ProcessInfo.processInfo.environment[key].flatMap { $0.isEmpty ? nil : $0 }
    }

    static func load() async throws -> (target: Qwen35, drafter: DFlash2DraftModel) {
        guard let targetPath = environment("DFLASH2_TARGET_DIR"),
            let drafterPath = environment("DFLASH2_DRAFTER_DIR")
        else { throw XCTSkip("set DFLASH2_TARGET_DIR and DFLASH2_DRAFTER_DIR") }
        if let loaded { return loaded }
        var configuration = ResolvedModelConfiguration(
            directory: URL(fileURLWithPath: targetPath))
        configuration.excludedComponents = [.vision]
        let context = try await VLMModelFactory.shared._load(
            configuration: configuration, tokenizerLoader: NoTokenizerLoader())
        let drafter = try await DFlash2DrafterModelFactory.load(
            from: URL(fileURLWithPath: drafterPath))
        let pair = (context.model as! Qwen35, drafter.model as! DFlash2DraftModel)
        loaded = pair
        return pair
    }

    /// Tokens per verify pass for the calibration and the speculative-decoding tests.
    static var block: Int { Int(environment("DFLASH2_BLOCK") ?? "") ?? 5 }

    static func prompts() throws -> [PromptFile] {
        guard let directory = environment("DFLASH2_TOKENS_DIR") else {
            throw XCTSkip("set DFLASH2_TOKENS_DIR")
        }
        guard let files = environment("DFLASH2_TOKEN_FILES") else {
            throw XCTSkip("set DFLASH2_TOKEN_FILES")
        }
        return try files.split(separator: ",").map(String.init).map {
            try JSONDecoder().decode(
                PromptFile.self,
                from: Data(
                    contentsOf: URL(fileURLWithPath: directory).appending(component: "\($0).json")
                ))
        }
    }
}

private func tokens(_ ids: [Int]) -> MLXArray { MLXArray(ids.map(Int32.init)) }

/// Prefill `ids` with the model's own `prepare`, as a prefix checkpoint is built.
private func prefill(_ model: Qwen35, _ ids: [Int]) throws -> (
    cache: [KVCache], state: LMOutput.State?, logits: MLXArray
) {
    let cache = try model.newCache(parameters: nil)
    guard
        case .logits(let output) = try model.prepare(
            LMInput(tokens: tokens(ids)), cache: cache, state: nil, prefill: .init())
    else { throw XCTSkip("Qwen35 prepare returns logits") }
    eval(output.logits, cache)
    return (cache, output.state, output.logits[0, -1])
}

private func copyCache(_ cache: [KVCache]) -> [KVCache] { cache.map { $0.copy() } }

/// Per-row distances between `[S, vocab]` logits: the largest difference among the
/// reference row's top 64 logits, in bf16 ulps of that row's top logit (an 8-bit
/// significand), and the mean absolute difference over the vocabulary.
private func distances(_ x: MLXArray, _ reference: MLXArray) -> (topUlps: MLXArray, mean: MLXArray)
{
    let x = x.asType(.float32)
    let reference = reference.asType(.float32)
    let top = argSort(-reference, axis: -1)[0..., ..<64]
    let difference = abs(takeAlong(x, top, axis: -1) - takeAlong(reference, top, axis: -1))
    let scale = abs(reference).max(axis: -1)
    let ulp = pow(MLXArray(Float(2)), floor(log2(scale)) - 7)
    return (difference.max(axis: -1) / ulp, abs(x - reference).mean(axis: -1))
}

/// Commit the first `count` rows of a verify pass at `position`, as a round does.
private func commit(
    _ cache: [KVCache], _ result: DFlash2VerifyResult, count: Int, position: Int
) {
    DFlash2TokenIterator.commitRecurrentState(
        cache: cache, captures: result.recurrentCaptures, validCount: MLXArray(Int32(count)))
    for case let attention as KVCacheSimple in cache {
        attention.commitRows(count: position + count)
    }
}

/// Keys and values of every attention layer at `rows`.
private func attentionRows(_ cache: [KVCache], _ rows: Range<Int>) -> [MLXArray] {
    cache.compactMap { $0 as? KVCacheSimple }.flatMap { entry in
        entry.state.map { $0[.ellipsis, rows, 0...] }
    }
}

/// Slot `slot` of every gated-delta layer: 0 the conv state, 1 the recurrent state.
private func recurrent(_ cache: [KVCache], _ slot: Int) -> [MLXArray] {
    cache.compactMap { ($0 as? MambaCache)?[slot] }
}

/// L2 norm of the difference over the reference's, across every array.
private func relativeError(_ x: [MLXArray], _ reference: [MLXArray]) -> MLXArray {
    let difference = zip(x, reference).map {
        square($0.asType(.float32) - $1.asType(.float32)).sum()
    }
    let scale = reference.map { square($0.asType(.float32)).sum() }
    return sqrt(stacked(difference).sum() / stacked(scale).sum())
}

private func summary(_ values: [Float]) -> (median: Float, p95: Float, max: Float) {
    let sorted = values.sorted()
    return (
        sorted[sorted.count / 2], sorted[Int((Double(sorted.count - 1) * 0.95).rounded())],
        sorted.last!
    )
}

/// V's error may be this multiple of F's. Clean runs at widths 5 and 8, native
/// and under `applegpu_g13s` routing, read at most 2.5 (half-block conv and
/// attention rows at width 8, where F is a forward of the half block).
private let teacherForcedFactor: Float = 3

/// Additive slack per quantity on top of the multiple of the forward's error,
/// for a quantity the forward reproduces exactly. Errors are top-64 ulps of the
/// top logit for logits, relative L2 for states.
private let teacherForcedFloors: [String: Float] = [
    "logits": 8,
    "attention rows": 0.002,
    "recurrent": 0.002,
    "conv": 0.002,
    "next logits": 8,
]

final class DFlash2RealWeightsTests: XCTestCase {

    /// Every other block of a greedy teacher-forced chain, a verify pass starts
    /// from the step chain's own state (S) and is compared with the steps over
    /// that block: its logits; after committing the whole block and after
    /// committing half of it, the committed attention rows, recurrent and conv
    /// states, and the next step's logits. An ordinary forward of the same
    /// width from the same state (F) sums in the same batched order the verify
    /// pass does, so its distance from S is the noise V cannot avoid; V's
    /// median and 95th-percentile error per quantity are bounded by F's.
    /// Breaks if verify positions, the masked scan, the replay or the commit
    /// count are off. The long chain (V and F committed block after block,
    /// never resynced) is printed as a diagnostic only.
    func testTeacherForcedVerifyTracksTheStep() async throws {
        let (model, _) = try await RealWeights.load()
        let prompt = try RealWeights.prompts()[0]
        let widths =
            RealWeights.environment("DFLASH2_TF_WIDTHS")?.split(separator: ",")
            .compactMap { Int($0) } ?? [5, 8]
        for width in widths {
            try teacherForced(model, prompt, width: width)
        }
    }

    private func teacherForced(_ model: Qwen35, _ prompt: PromptFile, width: Int) throws {
        let blocks = 1024 / width
        let partial = max(1, width / 2)

        let start = try prefill(model, prompt.tokenIds)
        let stepCache = start.cache
        var stepState = start.state
        let verifyChain = copyCache(stepCache)
        let forwardChain = copyCache(stepCache)
        var forwardChainState = start.state
        try model.dflash2Prepare(cache: verifyChain, state: start.state)

        var token = argMax(start.logits).item(Int.self)
        var position = prompt.tokenIds.count
        var chain: [(vsTop: Float, fsTop: Float, vsMean: Float, fsMean: Float)] = []
        var errors: [String: (verify: [Float], forward: [Float])] = [:]
        func record(_ name: String, _ verify: MLXArray, _ forward: MLXArray) {
            errors[name, default: ([], [])].verify.append(verify.item(Float.self))
            errors[name, default: ([], [])].forward.append(forward.item(Float.self))
        }

        for blockIndex in 0 ..< blocks {
            let sampled = blockIndex % 2 == 0
            let origin = sampled ? copyCache(stepCache) : []
            let originState = stepState
            var partialStep: [KVCache] = []
            var block: [Int] = []
            var stepRows: [MLXArray] = []
            for j in 0 ..< width {
                if sampled && j == partial { partialStep = copyCache(stepCache) }
                block.append(token)
                let output = model(
                    LMInput.Text(tokens: tokens([token]).reshaped(1, 1)), cache: stepCache,
                    state: stepState)
                stepState = output.state
                let row = output.logits[0, -1]
                eval(row)
                stepRows.append(row)
                token = argMax(row).item(Int.self)
            }
            let step = stacked(stepRows)

            // Long chain, never resynced.
            let chained = model.dflash2Verify(
                DFlash2VerifyRequest(
                    tokens: tokens(block).expandedDimensions(axis: 0),
                    position: MLXArray([Int32(position)]), positionUpperBound: position,
                    captureLayers: []),
                cache: verifyChain)
            commit(verifyChain, chained, count: width, position: position)
            let chainedForward = model(
                LMInput.Text(tokens: tokens(block).expandedDimensions(axis: 0)),
                cache: forwardChain, state: forwardChainState)
            forwardChainState = chainedForward.state
            let vs = distances(chained.logits[0], step)
            let fs = distances(chainedForward.logits[0], step)
            eval(
                vs.topUlps, vs.mean, fs.topUlps, fs.mean,
                (verifyChain + forwardChain).flatMap { $0.innerState() })
            let columns = [vs.topUlps, fs.topUlps, vs.mean, fs.mean].map { $0.asArray(Float.self) }
            for j in 0 ..< width {
                chain.append((columns[0][j], columns[1][j], columns[2][j], columns[3][j]))
            }

            if sampled {
                let next = model(
                    LMInput.Text(tokens: tokens([token]).reshaped(1, 1)),
                    cache: copyCache(stepCache), state: stepState
                ).logits[0, -1]
                for (count, reference, nextInput, nextReference) in [
                    (width, stepCache, token, next),
                    (partial, partialStep, block[partial], stepRows[partial]),
                ] {
                    let label = count == width ? "full" : "partial"
                    let verifyCache = copyCache(origin)
                    try model.dflash2Prepare(cache: verifyCache, state: originState)
                    let result = model.dflash2Verify(
                        DFlash2VerifyRequest(
                            tokens: tokens(block).expandedDimensions(axis: 0),
                            position: MLXArray([Int32(position)]), positionUpperBound: position,
                            captureLayers: []),
                        cache: verifyCache)
                    commit(verifyCache, result, count: count, position: position)
                    XCTAssertTrue(
                        verifyCache.filter { !($0 is MambaCache) }
                            .allSatisfy { $0.offset == position + count },
                        "width \(width) block \(blockIndex): \(label) commit offset")

                    let forwardCache = copyCache(origin)
                    let forward = model(
                        LMInput.Text(
                            tokens: tokens(Array(block[..<count])).expandedDimensions(axis: 0)),
                        cache: forwardCache, state: originState)

                    if count == width {
                        record(
                            "logits", distances(result.logits[0], step).topUlps.max(),
                            distances(forward.logits[0], step).topUlps.max())
                    }
                    let rows = position ..< (position + count)
                    record(
                        "attention rows \(label)",
                        relativeError(
                            attentionRows(verifyCache, rows), attentionRows(reference, rows)),
                        relativeError(
                            attentionRows(forwardCache, rows), attentionRows(reference, rows)))
                    record(
                        "recurrent \(label)",
                        relativeError(recurrent(verifyCache, 1), recurrent(reference, 1)),
                        relativeError(recurrent(forwardCache, 1), recurrent(reference, 1)))
                    record(
                        "conv \(label)",
                        relativeError(recurrent(verifyCache, 0), recurrent(reference, 0)),
                        relativeError(recurrent(forwardCache, 0), recurrent(reference, 0)))
                    let input = LMInput.Text(tokens: tokens([nextInput]).reshaped(1, 1))
                    let verifyNext = model(input, cache: verifyCache, state: originState)
                    let forwardNext = model(input, cache: forwardCache, state: forward.state)
                    record(
                        "next logits \(label)",
                        distances(verifyNext.logits[0], nextReference.expandedDimensions(axis: 0))
                            .topUlps.max(),
                        distances(forwardNext.logits[0], nextReference.expandedDimensions(axis: 0))
                            .topUlps.max())
                }
            }
            position += width
        }

        print(
            "teacher-forced \(prompt.label) width \(width): \(blocks) blocks after "
                + "\(prompt.tokenIds.count), every other one resynced")
        print(
            "quantity                 V median     V p95     V max   F median     F p95     F max")
        for (name, values) in errors.sorted(by: { $0.key < $1.key }) {
            let v = summary(values.verify)
            let f = summary(values.forward)
            print(
                String(
                    format: "%-22@ %10.4g %9.4g %9.4g %10.4g %9.4g %9.4g", name as NSString,
                    v.median, v.p95, v.max, f.median, f.p95, f.max))
        }
        print("long chain, top-64 error in ulps of the top logit and mean |d| over the vocabulary")
        print("bucket     V-S max  V-S mean  F-S max  F-S mean   V-S |d|   F-S |d|")
        for lower in stride(from: 0, to: chain.count, by: 256) {
            let slice = chain[lower ..< min(lower + 256, chain.count)]
            func average(_ values: [Float]) -> Float { values.reduce(0, +) / Float(values.count) }
            print(
                String(
                    format: "%4d-%4d  %7.1f  %8.2f  %7.1f  %8.2f  %8.4f  %8.4f", lower,
                    lower + slice.count - 1, slice.map(\.vsTop).max()!,
                    average(slice.map(\.vsTop)), slice.map(\.fsTop).max()!,
                    average(slice.map(\.fsTop)), average(slice.map(\.vsMean)),
                    average(slice.map(\.fsMean))))
        }

        for (name, values) in errors {
            let v = summary(values.verify)
            let f = summary(values.forward)
            let floor =
                teacherForcedFloors[
                    name.components(separatedBy: " ").dropLast().joined(
                        separator: " ")] ?? teacherForcedFloors[name]!
            XCTAssertLessThanOrEqual(
                v.median, teacherForcedFactor * f.median + floor,
                "width \(width) \(name): median verify error within a batched forward's")
            XCTAssertLessThanOrEqual(
                v.p95, teacherForcedFactor * f.p95 + floor,
                "width \(width) \(name): 95th-percentile verify error within a batched forward's")
        }
    }
    /// The attach-time calibration on the real pair: the round cost lies
    /// between one plain step and the block, the self-check passes on this
    /// GPU, and three calls, the first cold as at an attach, agree within 10%.
    /// Breaks if the calibration times something other than a full round or a
    /// plain step, or the self-check rejects a correct verify pass.
    func testCalibrationMeasuresAndChecksThisGPU() async throws {
        let (model, drafter) = try await RealWeights.load()
        let width = RealWeights.block
        let parameters = GenerateParameters(temperature: 0.15, topP: 0.8)
        var costs: [Double] = []
        for call in 0 ..< 3 {
            let start = Date.timeIntervalSinceReferenceDate
            let calibration = try dflash2Calibrate(
                target: model, drafter: drafter, blockSize: width, parameters: parameters)
            print(
                String(
                    format:
                        "calibration %d, block %d: R %.3f, plain %.2f ms, round %.2f ms, spread %.3f, %d measurement(s), %.2f s, %@",
                    call, width, calibration.roundCost, 1000 * calibration.plainStepSeconds,
                    1000 * calibration.roundSeconds, calibration.plainSpread,
                    calibration.measurements, Date.timeIntervalSinceReferenceDate - start,
                    "\(calibration.outcome)" as NSString))
            XCTAssertEqual(calibration.outcome, .passed, "call \(call)")
            XCTAssertGreaterThan(calibration.roundCost, 1, "call \(call)")
            XCTAssertLessThan(calibration.roundCost, 0.8 * Double(width), "call \(call)")
            costs.append(calibration.roundCost)
        }
        let spread = (costs.max()! - costs.min()!) / (costs.reduce(0, +) / Double(costs.count))
        XCTAssertLessThanOrEqual(spread, 0.10, "calls agree within 10%: \(costs)")
    }

    /// The attach-time self-check on the real pair passes a correct verify pass
    /// and fails one whose rows sit a position late or early, and a commit
    /// one row long or short, at widths 5 and 8. Breaks if the calibration
    /// prompt's continuation repeats or near-ties, so that a shifted row or
    /// commit still picks the plain step's token.
    func testSelfCheckCatchesShiftedRowsAndCommits() async throws {
        let (model, drafter) = try await RealWeights.load()
        let prefix = try DFlash2CalibrationPrefix(
            target: model, vocabularySize: model.dflash2Geometry.vocabularySize,
            parameters: GenerateParameters(temperature: 0.15, topP: 0.8))
        let plants: [(String, DFlash2SelfCheckFaults)] = [
            ("rows late", .init(verifyLogits: { concatenated([$0[1...], $0[(-1)...]]) })),
            (
                "rows early",
                .init(verifyLogits: { concatenated([$0[..<1], $0[..<($0.dim(0) - 1)]]) })
            ),
            ("commit long", .init(commitSkew: 1)),
            ("commit short", .init(commitSkew: -1)),
        ]
        for width in [5, 8] {
            func check(_ faults: DFlash2SelfCheckFaults) throws -> DFlash2SelfCheckFailure? {
                try dflash2SelfCheck(
                    target: model, prefix: prefix, blockSize: width,
                    captureLayers: drafter.targetLayerIds,
                    nearTieGap: DFlash2Calibration.nearTieGap, faults: faults)
            }
            XCTAssertNil(try check(.init()), "width \(width): clean")
            for (name, faults) in plants {
                let failure = try check(faults)
                print("self-check width \(width) \(name): \(String(describing: failure))")
                XCTAssertNotNil(failure, "width \(width): \(name) passes the self-check")
            }
        }
    }

    /// Two speculative turns from copies of one prefix checkpoint are
    /// identical, and the checkpoint is untouched. Breaks if the iterator
    /// writes through to the prefix cache it was given a copy of, or depends on
    /// state outside its inputs.
    func testConsecutiveTurnsFromOnePrefixAreIdentical() async throws {
        let (model, drafter) = try await RealWeights.load()
        let prompt = try RealWeights.prompts()[0].tokenIds
        let prefix = prompt.count - 512
        let tip = try prefill(model, Array(prompt[..<prefix]))
        let fingerprint = tip.cache.flatMap { $0.innerState() }.map {
            $0.asType(.float32).sum().item(Float.self)
        }
        var parameters = GenerateParameters(maxTokens: 256, temperature: 0.15, topP: 0.8)
        parameters.seed = 3

        func turn() throws -> (tokens: [Int], tau: Double) {
            var iterator = try DFlash2TokenIterator(
                input: LMInput(tokens: tokens(prompt)), mainModel: model, drafter: drafter,
                mainCache: copyCache(tip.cache), prefilledPrefixTokens: prefix, state: tip.state,
                parameters: parameters, blockSize: RealWeights.block, fallback: .disabled)
            var out: [Int] = []
            while let token = iterator.next() { out.append(token) }
            iterator.finalizeGeneration()
            let telemetry = iterator.speculativeDecodingTelemetry!
            return (
                out,
                Double(telemetry.acceptedDraftTokenCount + telemetry.roundCount)
                    / Double(telemetry.roundCount)
            )
        }
        let first = try turn()
        let second = try turn()
        print(
            "two turns, block \(RealWeights.block): \(first.tokens.count) tokens, "
                + "tau \(first.tau) / \(second.tau)")
        XCTAssertEqual(first.tokens, second.tokens)
        XCTAssertTrue(tip.cache.filter { !($0 is MambaCache) }.allSatisfy { $0.offset == prefix })
        XCTAssertEqual(
            tip.cache.flatMap { $0.innerState() }.map {
                $0.asType(.float32).sum().item(Float.self)
            }, fingerprint)
    }

    /// Greedy speculation equals greedy plain decoding until a near-tie. Prints
    /// the first divergence and the plain step's top-2 gap there.
    func testGreedySpeculationMatchesPlainUntilANearTie() async throws {
        let (model, drafter) = try await RealWeights.load()
        for prompt in try RealWeights.prompts() {
            let parameters = GenerateParameters(maxTokens: 400, temperature: 0)
            var plain = try TokenIterator(
                input: LMInput(tokens: tokens(prompt.tokenIds)), model: model,
                parameters: parameters)
            var expected: [Int] = []
            while let token = plain.next() { expected.append(token) }
            var iterator = try DFlash2TokenIterator(
                input: LMInput(tokens: tokens(prompt.tokenIds)), mainModel: model,
                drafter: drafter, parameters: parameters, blockSize: RealWeights.block,
                fallback: .disabled)
            var actual: [Int] = []
            while let token = iterator.next() { actual.append(token) }
            guard
                let divergence = zip(expected, actual).enumerated()
                    .first(where: { $0.element.0 != $0.element.1 })?.offset
            else {
                print(
                    "\(prompt.label), block \(RealWeights.block): greedy identical over "
                        + "\(actual.count) tokens")
                continue
            }
            // The plain step's top-2 gap where the two first differ.
            let held = try prefill(model, prompt.tokenIds + Array(expected[..<divergence]))
            let sorted = MLX.sorted(held.logits.asType(.float32)).asArray(Float.self)
            let gap = sorted[sorted.count - 1] - sorted[sorted.count - 2]
            print(
                "\(prompt.label), block \(RealWeights.block): greedy diverges at output "
                    + "\(divergence) of \(actual.count); "
                    + "plain top-2 gap \(gap)")
            XCTAssertLessThan(gap, 0.5, "\(prompt.label): divergence only at a near-tie")
        }
    }

    /// Sampled at the production temperature, speculative and plain sequences
    /// have the same mean target NLL: a permutation test over 50 independent
    /// samples per arm and packet. Breaks if acceptance biases what is emitted,
    /// as accepting a draft that is not the target's sample would.
    func testSampledNLLMatchesPlain() async throws {
        let (model, drafter) = try await RealWeights.load()
        let samples = Int(RealWeights.environment("DFLASH2_NLL_SAMPLES") ?? "") ?? 50
        let maxTokens = 128
        for prompt in try RealWeights.prompts() {
            let ids = prompt.tokenIds
            let prefix = ids.count - 512
            let tip = try prefill(model, Array(ids[..<prefix]))
            let full = try prefill(model, ids)

            /// Mean `-log p(token)` of `generated` under the raw target.
            func nll(_ generated: [Int]) -> Double {
                guard !generated.isEmpty else { return 0 }
                let cache = copyCache(full.cache)
                var rows = [full.logits.expandedDimensions(axis: 0)]
                if generated.count > 1 {
                    let output = model(
                        LMInput.Text(
                            tokens: tokens(Array(generated.dropLast())).expandedDimensions(axis: 0)),
                        cache: cache, state: full.state)
                    rows.append(output.logits[0])
                }
                let logprobs = logSoftmax(concatenated(rows).asType(.float32), axis: -1)
                let picked = takeAlong(
                    logprobs, tokens(generated).reshaped(-1, 1), axis: -1)
                return Double(-picked.mean().item(Float.self))
            }

            var plainNLL: [Double] = []
            var specNLL: [Double] = []
            var specTau: [Double] = []
            for sample in 0 ..< samples {
                var parameters = GenerateParameters(
                    maxTokens: maxTokens, temperature: 0.15, topP: 0.8)
                parameters.seed = UInt64(1000 + sample)
                var plain = try TokenIterator(
                    input: LMInput(tokens: tokens(Array(ids[prefix...]))), model: model,
                    cache: copyCache(tip.cache), state: tip.state, parameters: parameters)
                var plainOut: [Int] = []
                while let token = plain.next() { plainOut.append(token) }
                plainNLL.append(nll(plainOut))

                parameters.seed = UInt64(sample)
                var iterator = try DFlash2TokenIterator(
                    input: LMInput(tokens: tokens(ids)), mainModel: model, drafter: drafter,
                    mainCache: copyCache(tip.cache), prefilledPrefixTokens: prefix,
                    state: tip.state,
                    parameters: parameters, blockSize: RealWeights.block, fallback: .disabled)
                var specOut: [Int] = []
                while let token = iterator.next() { specOut.append(token) }
                specNLL.append(nll(specOut))
                let telemetry = iterator.speculativeDecodingTelemetry!
                specTau.append(
                    Double(telemetry.acceptedDraftTokenCount + telemetry.roundCount)
                        / Double(telemetry.roundCount))
            }

            let observed = abs(mean(specNLL) - mean(plainNLL))
            var generator = SplitMix(seed: 17)
            let pooled = specNLL + plainNLL
            let permutations = 10_000
            var extreme = 0
            for _ in 0 ..< permutations {
                var shuffled = pooled
                shuffled.shuffle(using: &generator)
                let a = mean(Array(shuffled[..<specNLL.count]))
                let b = mean(Array(shuffled[specNLL.count...]))
                if abs(a - b) >= observed { extreme += 1 }
            }
            let p = Double(extreme + 1) / Double(permutations + 1)
            print(
                String(
                    format:
                        "%@, block %d: mean NLL spec %.4f plain %.4f (n=%d each), p=%.3f, mean tau %.2f",
                    prompt.label as NSString, RealWeights.block, mean(specNLL), mean(plainNLL),
                    samples, p, mean(specTau)))
            XCTAssertGreaterThan(p, 0.01, "\(prompt.label): NLL differs between arms")
        }
    }
}

private func mean(_ values: [Double]) -> Double {
    values.reduce(0, +) / Double(max(values.count, 1))
}

private struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
