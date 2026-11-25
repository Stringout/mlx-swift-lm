// Synthetic tests for DFlash2 speculative decoding over the MLXVLM Qwen 3.5
// target. Drafter, context-cache, mask, conv and selector tests are ported from
// ml-explore/mlx-swift-lm#607 (spokvulcan). Tests that compare speculative output
// with plain decoding token for token run on the CPU stream: on M5 GPUs a
// multi-row float32 matmul runs as TF32 and a one-row matmul does not, so the
// GPU would compare different numerics rather than the algorithm.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

// MARK: - Fixtures

/// Layers `[linear, attention, linear, linear, attention]`.
private let tinySchedule = [
    "linear_attention", "full_attention", "linear_attention", "linear_attention", "full_attention",
]
private let vocabulary = 512
private let maskToken = 510
private let captureLayers = [1, 3]

/// A tiny Qwen 3.8-shaped VLM target. `linearHeadDim` 16 keeps the gated-delta
/// scan on the ops path, which the CPU runs; 32 takes the Metal kernel.
private func tinyTarget(
    seed: UInt64 = 1, dtype: DType = .float32, linearHeadDim: Int = 16
) throws -> Qwen35 {
    let layerTypes = tinySchedule.map { "\"\($0)\"" }.joined(separator: ", ")
    let json = """
        {
            "model_type": "qwen3_5",
            "image_token_id": 500,
            "video_token_id": 501,
            "vision_start_token_id": 502,
            "vision_end_token_id": 503,
            "vocab_size": \(vocabulary),
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": 5,
                "layer_types": [\(layerTypes)],
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": \(vocabulary),
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": \(linearHeadDim),
                "linear_value_head_dim": \(linearHeadDim),
                "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 4096,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "out_hidden_size": 64,
                "num_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 64
            }
        }
        """
    let config = try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
    let model = withRandomState(MLXRandom.RandomState(seed: seed)) { Qwen35(config) }
    if dtype != .float32 {
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
    }
    model.train(false)
    eval(model)
    return model
}

private func configJSON(
    hiddenSize: Int = 64, layers: Int = 2, heads: Int = 4, kvHeads: Int = 2,
    headDim: Int = 16, intermediate: Int = 96, vocab: Int = vocabulary, window: Int = 32,
    topK: Int = 4, rank: Int = 8, layerType: String = "sliding_attention",
    isCausal: String = "false", quantization: String = ""
) -> String {
    let layerTypes = Array(repeating: "\"\(layerType)\"", count: layers).joined(separator: ", ")
    return """
        {
          "architectures": ["DFlash2DraftModel"],
          "is_causal": \(isCausal),
          "dflash_config": {
            "block_size": 5,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": \(maskToken),
            "selector_rank": \(rank),
            "selector_top_k": \(topK),
            "target_layer_ids": [1, 3]
          },
          "head_dim": \(headDim),
          "hidden_size": \(hiddenSize),
          "intermediate_size": \(intermediate),
          "layer_types": [\(layerTypes)],
          "max_position_embeddings": 4096,
          "model_type": "qwen3",
          "num_attention_heads": \(heads),
          "num_hidden_layers": \(layers),
          "num_key_value_heads": \(kvHeads),
          "num_target_layers": 5,
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
          "sliding_window": \(window),
          \(quantization)
          "vocab_size": \(vocab)
        }
        """
}

private func tinyDrafterConfig(rank: Int = 8) -> DFlash2Configuration {
    try! JSONDecoder.json5().decode(
        DFlash2Configuration.self, from: Data(configJSON(rank: rank).utf8))
}

private func tinyDrafter(seed: UInt64 = 11, rank: Int = 8) -> DFlash2DraftModel {
    let drafter = withRandomState(MLXRandom.RandomState(seed: seed)) {
        DFlash2DraftModel(tinyDrafterConfig(rank: rank))
    }
    drafter.train(false)
    eval(drafter)
    return drafter
}

/// Drafts from a known continuation: the draft at absolute position `p` is
/// `reference[p]`, off by one where `wrongPositions` names `p`. Positions are
/// lazy in a pipelined round, so the lookup stays on the device.
private final class OracleDrafter: Module, DFlash2DrafterModel {
    let reference: MLXArray
    let wrong: MLXArray
    let count: Int

    init(reference: [Int], wrongPositions: Set<Int>) {
        self.count = reference.count
        self.reference = MLXArray(reference.map(Int32.init))
        self.wrong = MLXArray(reference.indices.map { wrongPositions.contains($0) })
        super.init()
    }

    var blockSize: Int { 5 }
    var maskTokenId: Int { maskToken }
    var targetLayerIds: [Int] { captureLayers }
    var contextWindow: Int { 31 }
    var targetGeometry: DFlash2Geometry {
        DFlash2Geometry(layerCount: 5, hiddenSize: 64, vocabularySize: vocabulary)
    }

    func makeState() -> DFlash2DrafterState { DFlash2DrafterState(contextCaches: []) }

    func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        let drafts = request.block.dim(1) - 1
        let anchor = Int32(request.contextPosition) + request.validRows.asType(.int32)
        let positions = minimum(
            anchor + 1 + MLXArray(Int32(0) ..< Int32(drafts)), MLXArray(Int32(count - 1)))
        let truth = take(reference, positions)
        let tokens = MLX.where(take(wrong, positions), (truth + 1) % Int32(vocabulary), truth)
        return DFlash2Proposal(
            tokens: tokens.expandedDimensions(axis: 0),
            candidates: tokens.expandedDimensions(axis: 0).expandedDimensions(axis: -1))
    }
}

private func promptTokens(_ count: Int, seed: Int = 0) -> [Int] {
    (0 ..< count).map { (($0 + seed) * 37 + 11) % 480 }
}

private func input(_ tokens: [Int]) -> LMInput {
    LMInput(tokens: MLXArray(tokens.map(Int32.init)))
}

private func drain(_ iterator: inout DFlash2TokenIterator, limit: Int = .max) -> [Int] {
    var tokens: [Int] = []
    while tokens.count < limit, let token = iterator.next() {
        tokens.append(token)
    }
    return tokens
}

/// Draft positions to get wrong so round `i`, anchored where the rounds before
/// it left off, emits `taus[i]` of a block of five.
private func wrongPositions(taus: [Int], first: Int) -> Set<Int> {
    var wrong = Set<Int>()
    var anchor = first
    for tau in taus {
        if tau < 5 { wrong.insert(anchor + tau) }
        anchor += tau
    }
    return wrong
}

/// One gated turn whose rounds emit `taus`, then right drafts; with the plain
/// tokens it must equal.
private func gatedTurn(
    _ model: Qwen35, _ prompt: [Int], taus: [Int], fallback: DFlash2Fallback
) throws -> (tokens: [Int], plain: [Int], rounds: Int, reason: String?) {
    let parameters = sampling(maxTokens: 1 + taus.reduce(0, +) + 20)
    let plain = try plainTokens(model, prompt, parameters)
    var iterator = try DFlash2TokenIterator(
        input: input(prompt), mainModel: model,
        drafter: OracleDrafter(
            reference: prompt + plain,
            wrongPositions: wrongPositions(taus: taus, first: prompt.count)),
        parameters: parameters, fallback: fallback)
    let tokens = drain(&iterator)
    return (
        tokens, plain, iterator.speculativeDecodingTelemetry?.roundCount ?? 0,
        iterator.passthroughReason
    )
}

private func onCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(.cpu, body)
}

private func sampling(maxTokens: Int, temperature: Float = 0.15, seed: UInt64 = 7)
    -> GenerateParameters
{
    var parameters = GenerateParameters(
        maxTokens: maxTokens, temperature: temperature, topP: 0.8,
        repetitionPenalty: 1.2, repetitionContextSize: 16)
    parameters.seed = seed
    return parameters
}

/// Plain decoding through the iterator's own prefill and per-position seeding.
private func plainTokens(
    _ model: Qwen35, _ prompt: [Int], _ parameters: GenerateParameters
) throws -> [Int] {
    var iterator = try DFlash2TokenIterator(
        input: input(prompt), mainModel: model,
        drafter: OracleDrafter(reference: prompt, wrongPositions: []), parameters: parameters,
        speculates: false)
    return drain(&iterator)
}

private func emitState() -> LMOutput.State {
    var state = LMOutput.State()
    state[mtpEmitFlagKey] = true
    state[mtpTargetLayerIdsKey] = captureLayers
    return state
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
}

private func bytes(_ array: MLXArray) -> Data {
    array.asData(access: .copy).data
}

/// Logits at the last of `tokens`, from one cold forward.
private func referenceLogits(_ model: Qwen35, _ tokens: [Int]) throws -> MLXArray {
    let cache = try model.newCache(parameters: nil)
    let output = model(
        LMInput.Text(tokens: MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)),
        cache: cache, state: nil)
    return output.logits[0, -1]
}

// MARK: - Configuration, cache, mask, conv, selector

final class DFlash2DrafterTests: XCTestCase {

    /// Breaks if a `config.json` key of the published checkpoint is misread.
    func testConfigurationDecodesReleaseCheckpoint() throws {
        // Mirrors z-lab/Qwen3.8-27B-DFlash2's config.json.
        let json = """
            {
              "architectures": ["DFlash2DraftModel"],
              "is_causal": false,
              "dflash_config": {
                "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61]
              },
              "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
              "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention"],
              "max_position_embeddings": 262144, "model_type": "qwen3",
              "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
              "num_target_layers": 64, "rms_norm_eps": 1e-06,
              "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" },
              "sliding_window": 2048, "tie_word_embeddings": false, "vocab_size": 248320,
              "quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
            }
            """
        let config = try JSONDecoder.json5().decode(
            DFlash2Configuration.self, from: Data(json.utf8))
        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.hiddenLayers, 5)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.ropeTheta, 10_000_000)
        XCTAssertEqual(config.slidingWindow, 2048)
        XCTAssertEqual(config.contextWindow, 2047)
        XCTAssertEqual(config.numTargetLayers, 64)
        XCTAssertEqual(config.dflash.blockSize, 8)
        XCTAssertEqual(config.dflash.maskTokenId, 248070)
        XCTAssertEqual(config.dflash.selectorTopK, 16)
        XCTAssertEqual(config.dflash.targetLayerIds, [5, 19, 33, 47, 61])
    }

    /// Breaks if a causal or full-attention drafter is accepted.
    func testConfigurationRejectsUnsupportedShapes() {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                DFlash2Configuration.self, from: Data(configJSON(isCausal: "true").utf8)))
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                DFlash2Configuration.self,
                from: Data(configJSON(layerType: "full_attention").utf8)))
    }

    /// Breaks if a drafter for another depth, width or vocabulary is accepted.
    func testGeometryChecksWidthVocabularyAndDepth() {
        let target = DFlash2Geometry(layerCount: 64, hiddenSize: 5120, vocabularySize: 248320)
        let ids = [5, 19, 33, 47, 61]
        func drafter(_ layers: Int?, _ hidden: Int = 5120, _ vocab: Int = 248320)
            -> DFlash2Geometry
        {
            DFlash2Geometry(layerCount: layers, hiddenSize: hidden, vocabularySize: vocab)
        }
        XCTAssertTrue(drafter(64).accepts(target, layerIds: ids))
        XCTAssertTrue(drafter(nil).accepts(target, layerIds: ids))
        XCTAssertFalse(drafter(nil).accepts(target, layerIds: ids + [64]))
        XCTAssertFalse(drafter(40).accepts(target, layerIds: ids))
        XCTAssertFalse(drafter(64, 2048).accepts(target, layerIds: ids))
        XCTAssertFalse(drafter(64, 5120, 202048).accepts(target, layerIds: ids))
    }

    /// Breaks if resolving marks the wrong rows valid.
    func testContextCachePlaceholdersResolve() {
        let cache = DFlash2ContextCache(window: 4)
        let rows = MLXArray.ones([1, 1, 3, 2])
        cache.append(keys: rows, values: rows, positions: [0, 1, 2])
        XCTAssertEqual(cache.rowValid, [false, false, false])
        cache.resolve(newest: 3, valid: 2)
        XCTAssertEqual(cache.rowValid, [true, true, false])
        let more = MLXArray.ones([1, 1, 2, 2]) * 2
        cache.append(keys: more, values: more, positions: [2, 3])
        cache.resolve(newest: 2, valid: 2)
        XCTAssertEqual(cache.count, 5)
        XCTAssertEqual(cache.rowPositions, [0, 1, 2, 2, 3])
        XCTAssertEqual(cache.rowValid, [true, true, false, true, true])
    }

    /// Breaks if compaction keeps placeholders or the wrong valid rows.
    func testContextCacheCompactionKeepsNewestValidRows() {
        let cache = DFlash2ContextCache(window: 4)
        var tag: Float = 0
        var position: Int32 = 0
        // 86 appends fill 258 of the 260-row buffer; the next one compacts.
        for _ in 0 ..< 86 {
            let rows = MLXArray.full([1, 1, 3, 2], values: MLXArray(tag))
            cache.append(
                keys: rows, values: rows, positions: [position, position + 1, position + 2])
            cache.resolve(newest: 3, valid: 2)
            tag += 1
            position += 3
        }
        let rows = MLXArray.full([1, 1, 3, 2], values: MLXArray(tag))
        cache.append(keys: rows, values: rows, positions: [position, position + 1, position + 2])
        XCTAssertEqual(cache.count, 7)
        XCTAssertEqual(cache.rowValid, [true, true, true, true, false, false, false])
        let values = cache.values![0, 0, 0..., 0].asArray(Float.self)
        XCTAssertEqual(values, [tag - 2, tag - 2, tag - 1, tag - 1, tag, tag, tag])
        XCTAssertEqual(Array(cache.rowPositions.suffix(3)), [position, position + 1, position + 2])
    }

    /// Breaks if the drafter sees placeholders or context past its window.
    func testVisibilityMaskWindowsAndHidesPlaceholders() {
        let cache = DFlash2ContextCache(window: 8)
        let old = MLXArray.ones([1, 1, 4, 2])
        cache.append(keys: old, values: old, positions: [0, 1, 2, 3])
        cache.resolve(newest: 4, valid: 4)
        let new = MLXArray.ones([1, 1, 2, 2])
        cache.append(keys: new, values: new, positions: [4, 5])
        let mask = DFlash2DraftModel.visibilityMask(
            cache: cache, appended: 2, validRows: MLXArray(Int32(1)),
            blockPosition: MLXArray([Int32(5)]), width: 3, window: 3)
        XCTAssertEqual(mask.shape, [3, 9])
        let expected: [[Bool]] = [
            [false, false, false, true, true, false, true, true, true],
            [false, false, false, false, true, false, true, true, true],
            [false, false, false, false, false, false, true, true, true],
        ]
        XCTAssertEqual(mask.asArray(Bool.self), expected.flatMap { $0 })
    }

    /// Breaks if a conv tap reads the wrong position or kernel slot.
    func testDynamicConvMatchesNaive() throws {
        let (hiddenSize, kernelSize, groupSize, length) = (16, 2, 4, 5)
        let conv = DFlash2DynamicConv(
            hiddenSize: hiddenSize, kernelSize: kernelSize, groupSize: groupSize)
        let (base, projection, hidden) = withRandomState(MLXRandom.RandomState(seed: 7)) {
            (
                MLXRandom.normal([2, kernelSize, hiddenSize]),
                MLXRandom.normal([2 * kernelSize * (hiddenSize / groupSize), hiddenSize]),
                MLXRandom.normal([1, length, hiddenSize])
            )
        }
        try conv.update(
            parameters: ModuleParameters.unflattened([
                ("base_kernel", base), ("kernel_projection.weight", projection),
            ]),
            verify: [])
        let (prepared, kernel) = conv.prepare(hidden)
        let finished = conv.finish(hidden, kernel: kernel)
        let dynamic = conv.kernelProjection(hidden).reshaped(
            1, length, 2, kernelSize, hiddenSize / groupSize)

        func naive(slot: Int, dynamic: [Float]) -> [Float] {
            let x = hidden.asArray(Float.self)
            let b = base.asArray(Float.self)
            var out = [Float](repeating: 0, count: length * hiddenSize)
            for t in 0 ..< length {
                for tap in 0 ..< kernelSize where t - tap >= 0 {
                    for channel in 0 ..< hiddenSize {
                        let group = channel / groupSize
                        let d = dynamic[(t * kernelSize + tap) * (hiddenSize / groupSize) + group]
                        let k = b[(slot * kernelSize + tap) * hiddenSize + channel] + d
                        out[t * hiddenSize + channel] += k * x[(t - tap) * hiddenSize + channel]
                    }
                }
            }
            return out
        }
        let slot0 = dynamic[0..., 0..., 0, 0..., 0...].asArray(Float.self)
        let slot1 = kernel.asArray(Float.self)
        for (actual, expected) in [
            (prepared.asArray(Float.self), naive(slot: 0, dynamic: slot0)),
            (finished.asArray(Float.self), naive(slot: 1, dynamic: slot1)),
        ] {
            for (a, e) in zip(actual, expected) {
                XCTAssertEqual(a, e, accuracy: 2e-2)
            }
        }
    }

    /// Breaks if the walk ignores the pairwise edges and takes each top-1.
    func testSelectorGreedyPrefersCoherentPath() throws {
        let config = try JSONDecoder.json5().decode(
            DFlash2Configuration.self,
            from: Data(
                configJSON(
                    hiddenSize: 8, layers: 1, heads: 1, kvHeads: 1, headDim: 8, intermediate: 8,
                    vocab: 12, window: 4, topK: 3, rank: 4
                ).utf8))
        let selector = DFlash2CandidateSelector(config)
        var projection = [Float](repeating: 0, count: 4 * 8)
        for i in 0 ..< 4 { projection[i * 8 + i] = 1 }
        var predecessor = [Float](repeating: 0, count: 12 * 4)
        predecessor[9 * 4 + 0] = 1
        var successor = [Float](repeating: 0, count: 12 * 4)
        successor[5 * 4 + 0] = 50
        try selector.update(
            parameters: ModuleParameters.unflattened([
                ("hidden_projection.weight", MLXArray(projection, [4, 8])),
                ("predecessor_codebook.weight", MLXArray(predecessor, [12, 4])),
                ("successor_codebook.weight", MLXArray(successor, [12, 4])),
            ]),
            verify: [])
        // Position 0: raw top-1 is 7, but 5 wins through the anchor edge; position 1: 2.
        var logits = [Float](repeating: -30, count: 2 * 12)
        logits[7] = 10
        logits[5] = 9
        logits[1] = 8
        logits[12 + 2] = 10
        logits[12 + 3] = 9
        logits[12 + 4] = 8
        for compiled in [false, true] {
            let proposal = selector.select(
                hidden: MLXArray.ones([1, 2, 8]), logits: MLXArray(logits, [1, 2, 12]),
                anchor: MLXArray([Int32(9)]), compiled: compiled)
            XCTAssertEqual(proposal.tokens.asArray(Int32.self), [5, 2])
            XCTAssertEqual(proposal.candidates.shape, [1, 2, 3])
        }
    }

    /// Breaks if a compiled segment reads stale weights, a different position,
    /// or if replacing modules leaves a trace behind.
    func testCompiledProposalMatchesEager() throws {
        let target = try tinyTarget()
        let compiled = tinyDrafter()
        let eager = tinyDrafter()
        eager.compiledTracesEnabled = false
        var stateA = compiled.makeState()
        var stateB = eager.makeState()
        let width = 2 * 64
        // A 20-row first context projects eagerly; later rounds trace.
        let rounds: [(rows: Int, valid: Int, width: Int)] = [
            (20, 20, 5), (3, 2, 5), (1, 1, 3), (5, 4, 5),
        ]
        var position = 0
        var anchor: Int32 = 7
        for (round, spec) in rounds.enumerated() {
            let hidden = withRandomState(MLXRandom.RandomState(seed: UInt64(round))) {
                MLXRandom.normal([1, spec.rows, width])
            }
            let block = MLXArray(
                [anchor] + Array(repeating: Int32(maskToken), count: spec.width - 1)
            ).expandedDimensions(axis: 0)
            let request = DFlash2ProposalRequest(
                block: block, targetHidden: hidden, contextPosition: position,
                validRows: MLXArray(Int32(spec.valid)))
            let a = compiled.propose(request, target: target, state: &stateA)
            let b = eager.propose(request, target: target, state: &stateB)
            XCTAssertEqual(a.tokens.asArray(Int32.self), b.tokens.asArray(Int32.self), "\(round)")
            XCTAssertEqual(
                a.candidates.asArray(Int32.self), b.candidates.asArray(Int32.self), "\(round)")
            for state in [stateA, stateB] {
                for cache in state.contextCaches {
                    cache.resolve(newest: spec.rows, valid: spec.valid)
                }
            }
            position += spec.valid
            anchor = a.tokens[0, 0].item(Int32.self)
        }
        XCTAssertGreaterThanOrEqual(compiled.compiledTraceCount, 6)
        XCTAssertEqual(eager.compiledTraceCount, 0)
        compiled.invalidateCompiledTraces()
        XCTAssertEqual(compiled.compiledTraceCount, 0)
    }

    /// Breaks if the drafter stops masking ids the target never emits.
    func testDrafterSkipsSuppressedTokens() throws {
        let target = try tinyTarget()
        let drafter = tinyDrafter(seed: 5)
        let allowed = Set(40 ..< 60)
        let request = DFlash2ProposalRequest(
            block: MLXArray([Int32(7)] + Array(repeating: Int32(maskToken), count: 4))
                .expandedDimensions(axis: 0),
            targetHidden: withRandomState(MLXRandom.RandomState(seed: 3)) {
                MLXRandom.normal([1, 6, 2 * 64])
            },
            contextPosition: 0, validRows: MLXArray(Int32(6)),
            suppressedTokens: SuppressTokensProcessor(
                tokenIds: Set(0 ..< vocabulary).subtracting(allowed)))
        var state = drafter.makeState()
        let proposal = drafter.propose(request, target: target, state: &state)
        XCTAssertTrue(
            Set(proposal.candidates.asArray(Int32.self).map(Int.init)).isSubset(of: allowed))
        XCTAssertTrue(Set(proposal.tokens.asArray(Int32.self).map(Int.init)).isSubset(of: allowed))
    }

    /// Breaks if the loader quantizes on load, ignores the artifact's
    /// `quantization` block, or renames the quantized codebooks.
    func testQuantizedArtifactLoadsAsStored() async throws {
        let target = try tinyTarget()
        let drafter = tinyDrafter(rank: 32)
        quantize(model: drafter, groupSize: 32, bits: 4)
        eval(drafter)
        XCTAssertTrue(drafter.candidateSelector.predecessorCodebook is QuantizedEmbedding)

        let directory = FileManager.default.temporaryDirectory.appending(
            component: "dflash2-q4-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try configJSON(
            rank: 32,
            quantization: #""quantization": {"group_size": 32, "bits": 4, "mode": "affine"},"#
        ).write(
            to: directory.appending(component: "config.json"), atomically: true, encoding: .utf8)
        let arrays = Dictionary(uniqueKeysWithValues: drafter.parameters().flattened())
        try save(
            arrays: arrays, metadata: ["format": "mlx"],
            url: directory.appending(component: "model.safetensors"))

        let container = try await DFlash2DrafterModelFactory.loadContainer(from: directory)
        let memory = try await container.memory()
        // Buffers are allocated in rounded sizes, so they hold at least the tensors.
        XCTAssertGreaterThanOrEqual(memory.model, arrays.values.reduce(0) { $0 + $1.nbytes })
        let loaded = await container.perform { SendableBox($0.model) }.consume()
        let codebook = (loaded as! DFlash2DraftModel).candidateSelector.predecessorCodebook
        XCTAssertTrue(codebook is QuantizedEmbedding)

        let request = DFlash2ProposalRequest(
            block: MLXArray([Int32(7)] + Array(repeating: Int32(maskToken), count: 4))
                .expandedDimensions(axis: 0),
            targetHidden: withRandomState(MLXRandom.RandomState(seed: 3)) {
                MLXRandom.normal([1, 6, 2 * 64])
            },
            contextPosition: 0, validRows: MLXArray(Int32(6)))
        var stateA = drafter.makeState()
        var stateB = loaded.makeState()
        let expected = drafter.propose(request, target: target, state: &stateA)
        let actual = loaded.propose(request, target: target, state: &stateB)
        XCTAssertEqual(actual.tokens.asArray(Int32.self), expected.tokens.asArray(Int32.self))
        XCTAssertEqual(
            actual.candidates.asArray(Int32.self), expected.candidates.asArray(Int32.self))
    }
}

// MARK: - Gated-delta capture and replay

final class DFlash2GatedDeltaTests: XCTestCase {

    /// Replaying `k` positions and scanning the rest from the replayed state
    /// reproduces the verify scan bit for bit, on a small layout and on
    /// Qwen3.8-27B's (16 key heads, 48 value heads, dim 128). Breaks if replay
    /// and verify take different scan routes, as an unmasked verify on the 27B
    /// layout would, or if replay keeps the wrong conv rows.
    func testReplayContinuesTheVerifyScanBitForBit() {
        let length = 5
        let kernelRows = 3
        for (hk, hv, d) in [(2, 4, 32), (16, 48, 128)] {
            let (q, k, v, a, b, aLog, dtBias, state, convInput) = withRandomState(
                MLXRandom.RandomState(seed: 11)
            ) {
                (
                    MLXRandom.normal([1, length, hk, d]).asType(.bfloat16) * 0.1,
                    MLXRandom.normal([1, length, hk, d]).asType(.bfloat16) * 0.1,
                    MLXRandom.normal([1, length, hv, d]).asType(.bfloat16),
                    MLXRandom.normal([1, length, hv]).asType(.bfloat16),
                    MLXRandom.normal([1, length, hv]).asType(.bfloat16),
                    MLXRandom.normal([hv]),
                    MLXRandom.normal([hv]),
                    MLXRandom.normal([1, hv, d, d]) * 0.1,
                    MLXRandom.normal([1, kernelRows + length, 2 * hk * d + hv * d])
                        .asType(.bfloat16)
                )
            }
            let (full, _) = gatedDeltaVerifyUpdate(
                q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state)
            let capture = GatedDeltaCapture(
                convInput: convInput, q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                initialState: state)
            for valid in 1 ..< length {
                // Lazy, as the iterator passes it.
                let (recurrent, conv) = capture.replay(
                    validCount: MLXArray(Int32(valid - 1)) + Int32(1))
                let (rest, _) = gatedDeltaVerifyUpdate(
                    q: q[0..., valid...], k: k[0..., valid...], v: v[0..., valid...],
                    a: a[0..., valid...], b: b[0..., valid...], aLog: aLog, dtBias: dtBias,
                    state: recurrent)
                XCTAssertEqual(
                    bytes(rest), bytes(full[0..., valid...]), "layout \(hk)/\(hv) valid \(valid)")
                XCTAssertEqual(
                    bytes(conv), bytes(convInput[0..., valid ..< (valid + kernelRows), 0...]))
            }
        }
    }
}

// MARK: - Qwen 3.5 VLM verify pass

final class DFlash2Qwen35VerifyTests: XCTestCase {

    /// The verify pass equals one forward over prompt and block, logits and
    /// captures alike, and committing any prefix then decoding matches that
    /// forward. Breaks if verify positions, attention mask, captures, replay or
    /// the commit are off.
    func testVerifyMatchesOneForwardAndCommitsAnyPrefix() throws {
        let model = try tinyTarget()
        XCTAssertEqual(
            model.languageModel.model.layers.map(\.isLinear),
            [
                true, false, true, true, false,
            ])
        let prompt = promptTokens(12)
        let block = [5, 11, 2, 40, 17]
        let tolerance = MatmulPrecision.splitTolerance

        let reference = model(
            LMInput.Text(
                tokens: MLXArray((prompt + block).map(Int32.init)).expandedDimensions(axis: 0)),
            cache: try model.newCache(parameters: nil), state: emitState())
        let referenceHidden = reference.state![mtpLastHiddenStatesKey]!
        XCTAssertEqual(referenceHidden.dim(-1), 2 * 64)

        for validCount in 1 ... block.count {
            let cache = try model.newCache(parameters: nil)
            let prefill = model(
                LMInput.Text(tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0)),
                cache: cache, state: emitState())
            eval(prefill.logits, cache)
            try model.dflash2Prepare(cache: cache, state: prefill.state)
            let result = model.dflash2Verify(
                DFlash2VerifyRequest(
                    tokens: MLXArray(block.map(Int32.init)).expandedDimensions(axis: 0),
                    position: MLXArray([Int32(prompt.count - 1)]) + 1,
                    positionUpperBound: prompt.count, captureLayers: captureLayers),
                cache: cache)
            let rows = prompt.count...
            XCTAssertLessThan(maxDifference(result.logits, reference.logits[0..., rows]), tolerance)
            XCTAssertLessThan(
                maxDifference(concatenated(result.hidden, axis: -1), referenceHidden[0..., rows]),
                tolerance)

            let committed = prompt.count + validCount
            for (entry, capture) in zip(
                cache.compactMap { $0 as? MambaCache }, result.recurrentCaptures)
            {
                let replayed = capture.replay(validCount: MLXArray(Int32(validCount)))
                entry[0] = replayed.conv
                entry[1] = replayed.recurrent
            }
            for case let attention as KVCacheSimple in cache {
                attention.commitRows(count: committed)
            }
            guard validCount < block.count else { continue }
            let next = model(
                LMInput.Text(tokens: MLXArray([Int32(block[validCount])]).reshaped(1, 1)),
                cache: cache, state: prefill.state)
            XCTAssertLessThan(
                maxDifference(next.logits[0, 0], reference.logits[0, committed]), tolerance,
                "validCount \(validCount)")
        }
    }

    /// The compiled verify segments equal the eager bodies bit for bit, and a
    /// module replacement drops them. Breaks if a trace omits a weight from its
    /// state, keys traces too coarsely, or survives a quantize.
    func testCompiledVerifyMatchesEagerAndFollowsModuleReplacement() throws {
        let model = try tinyTarget(dtype: .bfloat16, linearHeadDim: 32)
        let traces = model.languageModel.model.dflash2VerifyTraces
        let prompt = promptTokens(9)
        let block = [5, 11, 2, 40]

        func verify(compiled: Bool) throws -> [Data] {
            traces.compiledEnabled = compiled
            defer { traces.compiledEnabled = true }
            let cache = try model.newCache(parameters: nil)
            let prefill = model(
                LMInput.Text(tokens: MLXArray(prompt.map(Int32.init)).expandedDimensions(axis: 0)),
                cache: cache, state: nil)
            eval(prefill.logits, cache)
            try model.dflash2Prepare(cache: cache, state: prefill.state)
            let result = model.dflash2Verify(
                DFlash2VerifyRequest(
                    tokens: MLXArray(block.map(Int32.init)).expandedDimensions(axis: 0),
                    position: MLXArray([Int32(prompt.count)]), positionUpperBound: prompt.count,
                    captureLayers: captureLayers),
                cache: cache)
            return ([result.logits] + result.hidden + result.recurrentCaptures.map(\.q)).map(bytes)
        }

        XCTAssertEqual(try verify(compiled: true), try verify(compiled: false))
        XCTAssertGreaterThan(traces.traces.count, 0)

        quantize(model: model, groupSize: 32, bits: 4)
        try model.prepare()
        eval(model)
        let quantizedEager = try verify(compiled: false)
        XCTAssertEqual(traces.traces.count, 0, "dflash2Prepare drops traces of replaced modules")
        XCTAssertEqual(try verify(compiled: true), quantizedEager)
        XCTAssertGreaterThan(traces.traces.count, 0)
    }
}

// MARK: - Iterator

final class DFlash2IteratorTests: XCTestCase {

    /// With drafts forced wrong at chosen positions, sample-and-match emits
    /// exactly what plain seeded decoding emits at temperature 0.15 with top-p
    /// 0.8 and a repetition penalty, with the oracle and with a real drafter.
    /// Breaks if a verify row is sampled with another position's key or a
    /// processor history the plain path would not have, if the bonus is taken
    /// from the wrong row, or if a commit keeps rejected state.
    func testSampleAndMatchEqualsPlainSeededDecoding() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(40)
            let parameters = sampling(maxTokens: 48)
            let plain = try plainTokens(model, prompt, parameters)
            XCTAssertEqual(plain.count, 48)
            XCTAssertNotEqual(
                plain, try plainTokens(model, prompt, sampling(maxTokens: 48, seed: 8)),
                "the seed must matter, or this test does not exercise sampling")

            // Wrong in the first draft slot, a run across rounds, and a last slot.
            let wrong: Set<Int> = [41, 50, 51, 52, 60, 64, 77]
            var oracle = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model,
                drafter: OracleDrafter(reference: prompt + plain, wrongPositions: wrong),
                parameters: parameters, fallback: .disabled)
            XCTAssertEqual(drain(&oracle), plain)
            XCTAssertGreaterThan(oracle.acceptedCount, 0)
            XCTAssertLessThan(oracle.acceptedCount, oracle.proposedCount)

            let drafter = tinyDrafter()
            drafter.compiledTracesEnabled = false
            var real = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter,
                parameters: parameters, fallback: .disabled)
            XCTAssertEqual(drain(&real), plain)
            XCTAssertGreaterThan(real.speculativeDecodingTelemetry?.roundCount ?? 0, 0)
        }
    }

    /// Plain mode is ordinary greedy decoding. Breaks if the capture prefill
    /// or the plain step diverges from `TokenIterator`.
    func testPlainModeMatchesTokenIteratorGreedy() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(30)
            let parameters = GenerateParameters(maxTokens: 24, temperature: 0)
            var upstream = try TokenIterator(
                input: input(prompt), model: model, parameters: parameters)
            var expected: [Int] = []
            while let token = upstream.next() { expected.append(token) }
            XCTAssertEqual(try plainTokens(model, prompt, parameters), expected)
        }
    }

    /// The turn falls back once the running loss reaches the budget, decided on
    /// the rounds already synced, so the round in flight still commits, and the
    /// output stays plain. All-wrong drafts emit one token per round, so each
    /// round adds `R − 1` to the loss. Breaks if the trip compares with `>`
    /// instead of `>=`, counts the loss from another quantity, wastes or skips
    /// the in-flight round, or loses a token at the switch.
    func testLossBudgetTripsWhenTheLossReachesTheBudget() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(40)
            let parameters = sampling(maxTokens: 40)
            let plain = try plainTokens(model, prompt, parameters)
            let reference = prompt + plain

            // R 2.5 loses 1.5 per round: the loss reaches 6 at round 4 and 6.1 at round 5.
            for (budget, rounds) in [(6.0, 5), (6.1, 6)] {
                var wrong = try DFlash2TokenIterator(
                    input: input(prompt), mainModel: model,
                    drafter: OracleDrafter(
                        reference: reference, wrongPositions: Set(reference.indices)),
                    parameters: parameters,
                    fallback: DFlash2Fallback(roundCost: 2.5, lossBudget: budget))
                XCTAssertEqual(drain(&wrong), plain, "budget \(budget)")
                XCTAssertEqual(
                    wrong.speculativeDecodingTelemetry?.roundCount, rounds, "budget \(budget)")
                XCTAssertEqual(wrong.speculationLoss, 1.5 * Double(rounds), "budget \(budget)")
                XCTAssertNotNil(wrong.passthroughReason)
            }

            // Right drafts emit five tokens per round, more than any round cost below five.
            var right = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model,
                drafter: OracleDrafter(reference: reference, wrongPositions: []),
                parameters: parameters, fallback: DFlash2Fallback(roundCost: 4.9, lossBudget: 0.1))
            XCTAssertEqual(drain(&right), plain)
            XCTAssertNil(right.passthroughReason)
            XCTAssertEqual(right.speculationLoss, 0)
        }
    }

    /// The loss floors at zero: a winning stretch pays back an earlier loss, and
    /// a loss after a long winning stretch trips as soon as it would from a
    /// fresh start. Breaks if wins are ignored (the first turn trips) or if
    /// wins bank credit below zero (the second trips late).
    func testLossBudgetForgivesALossAWinningStretchRepaid() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(40)
            let fallback = DFlash2Fallback(roundCost: 3, lossBudget: 5)

            // Loses 4, wins it back, loses 4 again: never 5 at once.
            let repaid = [1, 1, 5, 5, 1, 1]
            let first = try gatedTurn(model, prompt, taus: repaid, fallback: fallback)
            XCTAssertEqual(first.tokens, first.plain)
            XCTAssertNil(first.reason)

            // Ten wins bank nothing: three losses of 2 reach 6 at the third, which
            // the round after it still commits.
            let late = Array(repeating: 5, count: 10) + [1, 1, 1, 1, 1, 1]
            let second = try gatedTurn(model, prompt, taus: late, fallback: fallback)
            XCTAssertEqual(second.tokens, second.plain)
            XCTAssertNotNil(second.reason)
            XCTAssertEqual(second.rounds, 14)
        }
    }

    /// Once fallen back, a turn never speculates again, however well the
    /// drafter would do. Breaks if the passthrough is reset or rounds resume.
    func testLossBudgetFallbackIsOneWay() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(40)
            let taus = [1, 1] + Array(repeating: 5, count: 20)
            let turn = try gatedTurn(
                model, prompt, taus: taus, fallback: DFlash2Fallback(roundCost: 2, lossBudget: 2))
            XCTAssertEqual(turn.tokens, turn.plain)
            XCTAssertNotNil(turn.reason)
            XCTAssertEqual(turn.rounds, 3, "two losing rounds trip; the third was in flight")
        }
    }

    /// Over round costs 1.5–4.5, two budgets and constant and noisy acceptance,
    /// the turn trips exactly one round after the loss first reaches the budget
    /// and loses at most the budget plus the crossing round and the round in
    /// flight, each at most `R − 1`. Breaks if the loss or the trip is computed
    /// differently from `S = max(0, S + R − τ)`, `S >= c`.
    func testLossBudgetBoundsTheLossOverAGrid() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(40)
            var generator: UInt64 = 5
            let noisy = (0 ..< 30).map { _ -> Int in
                generator = generator &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                return [1, 1, 1, 2, 2, 3, 5][Int(generator >> 33) % 7]
            }
            let traces = [
                ("ones", Array(repeating: 1, count: 30)),
                ("twos", Array(repeating: 2, count: 30)),
                ("noisy", noisy),
            ]
            for roundCost in [1.5, 2.5, 3.5, 4.5] {
                for budget in [3.0, 6.0] {
                    for (name, taus) in traces {
                        let label = "R \(roundCost) c \(budget) \(name)"
                        let fallback = DFlash2Fallback(roundCost: roundCost, lossBudget: budget)
                        let turn = try gatedTurn(model, prompt, taus: taus, fallback: fallback)
                        XCTAssertEqual(turn.tokens, turn.plain, label)

                        var loss = 0.0
                        var tripRound: Int?
                        for (index, tau) in taus.enumerated() {
                            loss = max(0, loss + roundCost - Double(tau))
                            if loss >= budget {
                                tripRound = index
                                break
                            }
                        }
                        if let tripRound {
                            XCTAssertEqual(turn.rounds, tripRound + 2, label)
                            let emitted = taus.prefix(tripRound + 2).reduce(0, +)
                            let lost = roundCost * Double(tripRound + 2) - Double(emitted)
                            XCTAssertLessThanOrEqual(lost, budget + 2 * (roundCost - 1), label)
                        } else {
                            XCTAssertNil(turn.reason, label)
                        }
                    }
                }
            }
        }
    }
    /// Speculating over a warm prefix equals speculating cold over the whole
    /// prompt. Breaks if the warm path prefills from the wrong offset, drops
    /// the carried state, or positions the suffix wrongly.
    func testWarmStartMatchesCold() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(40)
            let prefix = 25
            let parameters = sampling(maxTokens: 32)
            let drafter = tinyDrafter()
            drafter.compiledTracesEnabled = false

            var cold = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter,
                parameters: parameters, fallback: .disabled)
            let coldTokens = drain(&cold)

            let cache = try model.newCache(parameters: nil)
            guard
                case .logits(let output) = try model.prepare(
                    input(Array(prompt[..<prefix])), cache: cache, state: nil,
                    prefill: .init())
            else { return XCTFail("Qwen35 prepare returns logits") }
            eval(cache)
            var warm = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter, mainCache: cache,
                prefilledPrefixTokens: prefix, state: output.state, parameters: parameters,
                fallback: .disabled)
            XCTAssertEqual(drain(&warm), coldTokens)
        }
    }

    /// Finalizing after any number of drained tokens leaves the cache at the
    /// drained prefix, and decoding from it matches a cold forward. Breaks if
    /// the rewind trims the wrong count, skips the recurrent replay, or leaves
    /// the layers at different offsets.
    func testFinalizeRewindsUndrainedDrafts() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(30)
            let parameters = sampling(maxTokens: 30)
            let plain = try plainTokens(model, prompt, parameters)
            let reference = prompt + plain

            for drained in [1, 2, 3, 5, 6, 8] {
                var iterator = try DFlash2TokenIterator(
                    input: input(prompt), mainModel: model,
                    drafter: OracleDrafter(reference: reference, wrongPositions: [37]),
                    parameters: parameters, fallback: .disabled)
                XCTAssertEqual(drain(&iterator, limit: drained), Array(plain[..<drained]))
                iterator.finalizeGeneration()
                let held = iterator.anchorPosition
                XCTAssertTrue(
                    (prompt.count + drained - 1 ... prompt.count + drained).contains(held),
                    "drained \(drained): cache holds \(held)")
                XCTAssertTrue(
                    iterator.cache.filter { !($0 is MambaCache) }.allSatisfy { $0.offset == held })

                let next = model(
                    LMInput.Text(tokens: MLXArray([Int32(reference[held])]).reshaped(1, 1)),
                    cache: iterator.cache, state: iterator.state)
                XCTAssertLessThan(
                    maxDifference(
                        next.logits[0, 0], try referenceLogits(model, Array(reference[...held]))),
                    1e-3, "drained \(drained)")
            }
        }
    }

    /// The state the iterator returns continues a warm `prepare` on its cache,
    /// without the drafter exchange keys. Breaks if the state loses the rope
    /// delta or keeps the emit flag.
    func testReturnedStateContinuesAWarmPrepare() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(30)
            let drafter = tinyDrafter()
            drafter.compiledTracesEnabled = false
            var iterator = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter,
                parameters: sampling(maxTokens: 20), fallback: .disabled)
            let generated = drain(&iterator)
            iterator.finalizeGeneration()
            XCTAssertNil(iterator.state?[mtpEmitFlagKey])
            XCTAssertNil(iterator.state?[mtpLastHiddenStatesKey])
            XCTAssertNotNil(iterator.state?[LMOutput.Key<MLXArray>("qwen35.ropeDeltas")])

            let held = Array((prompt + generated)[..<iterator.anchorPosition])
            let suffix = promptTokens(7, seed: 3)
            guard
                case .logits(let output) = try model.prepare(
                    input(suffix), cache: iterator.cache, state: iterator.state,
                    prefill: .init())
            else { return XCTFail("Qwen35 prepare returns logits") }
            XCTAssertLessThan(
                maxDifference(output.logits[0, -1], try referenceLogits(model, held + suffix)),
                1e-3)
        }
    }

    /// Every unsupported input throws before prefill and leaves the cache as it
    /// was. Breaks if a check is dropped or moved after the prefill.
    func testUnsupportedInputsThrowBeforePrefill() throws {
        let model = try tinyTarget()
        let prompt = promptTokens(20)
        let drafter = OracleDrafter(reference: prompt, wrongPositions: [])
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0)

        func expect(
            _ expected: (DFlash2SpeculationError) -> Bool, _ message: String,
            _ make: () throws -> DFlash2TokenIterator
        ) {
            XCTAssertThrowsError(try make(), message) { error in
                guard let error = error as? DFlash2SpeculationError, expected(error) else {
                    return XCTFail("\(message): unexpected \(error)")
                }
            }
        }

        var delta = LMOutput.State()
        delta[LMOutput.Key<MLXArray>("qwen35.ropeDeltas")] = MLXArray([Int32(5)])
        expect({ if case .unsupportedPositions = $0 { true } else { false } }, "rope delta") {
            try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter, state: delta,
                parameters: parameters, fallback: .disabled)
        }

        let image = LMInput(
            text: .init(tokens: MLXArray(prompt.map(Int32.init))),
            image: .init(pixels: MLXArray.zeros([4, 4])))
        expect({ $0 == .textOnly }, "media") {
            try DFlash2TokenIterator(
                input: image, mainModel: model, drafter: drafter, parameters: parameters,
                fallback: .disabled)
        }

        var quantized = parameters
        quantized.kvBits = 4
        expect({ if case .unsupportedCache = $0 { true } else { false } }, "kv plan") {
            try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter, parameters: quantized,
                fallback: .disabled)
        }
        for (name, entry) in [
            ("quantized", QuantizedKVCache(groupSize: 32, bits: 8) as KVCache),
            ("rotating", RotatingKVCache(maxSize: 8)),
        ] {
            var cache = try model.newCache(parameters: nil)
            cache[1] = entry
            expect({ if case .unsupportedCache = $0 { true } else { false } }, name) {
                try DFlash2TokenIterator(
                    input: input(prompt), mainModel: model, drafter: drafter, mainCache: cache,
                    parameters: parameters, fallback: .disabled)
            }
        }

        let warm = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input(Array(prompt[..<10])), cache: warm, state: nil, prefill: .init())
        else { return XCTFail("Qwen35 prepare returns logits") }
        eval(warm)
        expect({ if case .missingContinuationState = $0 { true } else { false } }, "no state") {
            try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter, mainCache: warm,
                prefilledPrefixTokens: 10, parameters: parameters, fallback: .disabled)
        }
        expect({ if case .cacheOffsetMismatch = $0 { true } else { false } }, "offset") {
            try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter, mainCache: warm,
                prefilledPrefixTokens: 9, state: output.state, parameters: parameters,
                fallback: .disabled)
        }
        expect({ if case .promptTooShort = $0 { true } else { false } }, "prefix") {
            try DFlash2TokenIterator(
                input: input(Array(prompt[..<10])), mainModel: model, drafter: drafter,
                mainCache: warm, prefilledPrefixTokens: 10, state: output.state,
                parameters: parameters, fallback: .disabled)
        }
        expect({ $0 == .invalidBlockSize(1) }, "block") {
            try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter, parameters: parameters,
                blockSize: 1, fallback: .disabled)
        }
        for roundCost in [5.0, Double.nan] {
            expect(
                { if case .roundCostCannotPay = $0 { true } else { false } },
                "round cost \(roundCost)"
            ) {
                try DFlash2TokenIterator(
                    input: input(prompt), mainModel: model, drafter: drafter, mainCache: warm,
                    prefilledPrefixTokens: 10, state: output.state, parameters: parameters,
                    blockSize: 5, fallback: DFlash2Fallback(roundCost: roundCost, blockSize: 5))
            }
        }
        XCTAssertNoThrow(
            try DFlash2TokenIterator(
                input: input(prompt), mainModel: model, drafter: drafter,
                parameters: parameters, blockSize: 5,
                fallback: DFlash2Fallback(roundCost: 4.99, blockSize: 5)))
        XCTAssertTrue(
            warm.filter { !($0 is MambaCache) }.allSatisfy { $0.offset == 10 },
            "a refused stream leaves the cache")
    }
}

// MARK: - generate

private struct CountingTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens _: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(repeating: "x", count: tokenIds.count)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "x" }
    func applyChatTemplate(
        messages _: [[String: any Sendable]], tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct NoInputProcessor: UserInputProcessor {
    func prepare(input _: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([1])) }
}

final class DFlash2GenerateTests: XCTestCase {

    /// The stream reports the speculation stats and the prefix it reused, and
    /// its task ends after the info event. Breaks if the stats conformance or
    /// the prompt accounting is dropped.
    func testGenerateReportsSpeculationAndPrefix() async throws {
        let model = try tinyTarget()
        let prompt = promptTokens(30)
        let prefix = 20
        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input(Array(prompt[..<prefix])), cache: cache, state: nil, prefill: .init())
        else { return XCTFail("Qwen35 prepare returns logits") }
        eval(cache)
        let context = ModelContext(
            configuration: ModelConfiguration(id: "dflash2-test"), model: model,
            processor: NoInputProcessor(), tokenizer: CountingTokenizer())
        let (stream, task) = try generate(
            input: input(prompt), cache: cache, prefilledPrefixTokens: prefix,
            state: output.state, parameters: GenerateParameters(maxTokens: 16, temperature: 0),
            context: context,
            dflash2Drafter: OracleDrafter(reference: prompt, wrongPositions: []),
            blockSize: 5, fallback: .disabled)
        var info: GenerateCompletionInfo?
        for await generation in stream {
            if case .info(let value) = generation { info = value }
        }
        await task.value
        let completion = try XCTUnwrap(info)
        XCTAssertEqual(completion.generationTokenCount, 16)
        XCTAssertEqual(completion.promptTokenCount, prompt.count - prefix)
        XCTAssertEqual(completion.cachedPromptTokenCount, prefix)
        XCTAssertGreaterThan(completion.proposedDraftTokens ?? 0, 0)
        XCTAssertNotNil(completion.acceptedDraftTokens)
        XCTAssertNil(completion.passthroughReason)
        XCTAssertGreaterThan(completion.speculativeDecodingTelemetry?.roundCount ?? 0, 0)
    }
}

// MARK: - Lifetime

final class DFlash2LifetimeTests: XCTestCase {

    /// After a speculative generation over the compiled verify and drafter
    /// traces, dropping the target and drafter frees them and their weights.
    /// Breaks if anything a turn creates (verify traces, drafter traces, the
    /// iterator) keeps the model or its weight arrays alive.
    func testSpeculativeTurnReleasesTargetAndDrafter() throws {
        weak var weakTarget: Qwen35?
        weak var weakDrafter: DFlash2DraftModel?
        weak var weakLayerWeight: MLXArray?
        weak var weakDrafterWeight: MLXArray?
        try autoreleasepool {
            let target = try tinyTarget(dtype: .bfloat16, linearHeadDim: 32)
            let drafter = tinyDrafter()
            drafter.update(parameters: drafter.parameters().mapValues { $0.asType(.bfloat16) })
            eval(drafter)
            weakTarget = target
            weakDrafter = drafter
            weakLayerWeight = target.languageModel.model.layers[2].linearAttn!.outProj.weight
            weakDrafterWeight = drafter.layers[0].mlp.downProj.weight
            var iterator = try DFlash2TokenIterator(
                input: input(promptTokens(20)), mainModel: target, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 24, temperature: 0),
                fallback: .disabled)
            XCTAssertEqual(drain(&iterator).count, 24)
            iterator.finalizeGeneration()
            XCTAssertGreaterThan(target.languageModel.model.dflash2VerifyTraces.traces.count, 0)
            XCTAssertGreaterThan(drafter.compiledTraceCount, 0)
            Stream.gpu.synchronize()
        }
        XCTAssertNil(weakTarget, "the target model outlives its last reference")
        XCTAssertNil(weakDrafter, "the drafter outlives its last reference")
        XCTAssertNil(weakLayerWeight, "a target weight array outlives the target")
        XCTAssertNil(weakDrafterWeight, "a drafter weight array outlives the drafter")
    }
}

final class DFlash2PlainRouteTests: XCTestCase {

    /// Plain steps run as one-row verify passes decode what ordinary steps
    /// decode, so the bench's verify-route arm isolates numerics. Breaks if a
    /// one-row verify commit keeps the wrong recurrent state or offset.
    func testVerifyRoutePlainStepsMatchOrdinarySteps() throws {
        try onCPU {
            let model = try tinyTarget()
            let prompt = promptTokens(30)
            let parameters = sampling(maxTokens: 32)
            var verifyRoute = try DFlash2TokenIterator(
                input: input(prompt), mainModel: model,
                drafter: OracleDrafter(reference: prompt, wrongPositions: []),
                parameters: parameters, speculates: false, plainStepsVerify: true)
            XCTAssertEqual(drain(&verifyRoute), try plainTokens(model, prompt, parameters))
        }
    }
}

// MARK: - Calibration

final class DFlash2CalibrationTests: XCTestCase {

    /// Clock stamps per measurement: plain steps, rounds, plain steps.
    private let plainStamps = DFlash2Calibration.plainIntervals + 1
    private let roundStamps = DFlash2Calibration.roundIntervals + 1
    private var stampsPerMeasurement: Int { 2 * plainStamps + roundStamps }

    /// A clock that advances by `increment(call)` on each read.
    private final class ScriptedClock {
        var now = 0.0
        var calls = 0
        let increment: (Int) -> Double
        init(_ increment: @escaping (Int) -> Double) { self.increment = increment }
        func read() -> Double {
            now += increment(calls)
            calls += 1
            return now
        }
    }

    private func calibrate(_ model: Qwen35, clock: ScriptedClock) throws -> DFlash2Calibration {
        let prompt = promptTokens(40)
        return try dflash2Calibrate(
            target: model, drafter: OracleDrafter(reference: prompt, wrongPositions: []),
            blockSize: 5, parameters: sampling(maxTokens: 4), clock: clock.read)
    }

    /// The round cost is the median round over the mean of the two plain
    /// medians, and the spread is the larger plain MAD over median. Breaks if
    /// means replace medians (the outliers move them) or a phase is dropped.
    func testSummaryTakesMediansAndTheirRatio() {
        let summary = DFlash2Calibration.summarize(
            before: [1.0, 1.1, 0.9, 1.0, 9.0, 1.0, 1.0, 1.0],
            rounds: [2.0, 2.2, 1.8, 30.0, 2.0, 2.1, 1.9, 2.0],
            after: [3.0, 3.0, 2.7, 3.0, 3.3, 3.0, 0.1, 3.0])
        XCTAssertEqual(summary.plainStepSeconds, 2.0, accuracy: 1e-12)
        XCTAssertEqual(summary.roundSeconds, 2.0, accuracy: 1e-12)
        XCTAssertEqual(summary.roundCost, 1.0, accuracy: 1e-12)
        XCTAssertEqual(summary.plainSpread, 0.0, accuracy: 1e-12)
        XCTAssertEqual(
            DFlash2Calibration.summarize(
                before: [1, 1, 2, 2], rounds: [3], after: [1, 1, 1, 1]
            ).plainSpread, 1.0 / 3.0, accuracy: 1e-12)
    }

    /// Under a drift of 0.5% per plain step, linear in time, bracketing the
    /// rounds with plain steps recovers the true cost within 1%, where the
    /// plain steps before alone read it 7% high. Breaks if either plain phase
    /// is dropped from the denominator.
    func testSummaryCancelsLinearDrift() {
        let trueCost = 2.5
        let drift = 0.005
        var time = 0.0
        // Each interval runs at the rate of its own midpoint.
        func intervals(_ count: Int, scale: Double) -> [Double] {
            (0 ..< count).map { _ in
                let interval = scale * (1 + drift * time) / (1 - drift * scale / 2)
                time += interval
                return interval
            }
        }
        let before = intervals(8, scale: 1)
        let rounds = intervals(8, scale: trueCost)
        let after = intervals(8, scale: 1)
        let summary = DFlash2Calibration.summarize(before: before, rounds: rounds, after: after)
        XCTAssertEqual(summary.roundCost, trueCost, accuracy: 0.01 * trueCost)
        XCTAssertGreaterThan(
            DFlash2Calibration.median(rounds) / DFlash2Calibration.median(before),
            1.07 * trueCost)
    }

    /// Through the real phases on a tiny target, the clock's plain and round
    /// intervals give the round cost, from one measurement. Breaks if stamps
    /// land in the wrong phase or a phase reads the clock a different number
    /// of times.
    func testCalibrationTimesPlainStepsAndRoundsApart() throws {
        try onCPU {
            let model = try tinyTarget()
            let (plainStamps, roundStamps) = (self.plainStamps, self.roundStamps)
            let clock = ScriptedClock { call in
                let index = call % (2 * plainStamps + roundStamps)
                return (plainStamps ..< plainStamps + roundStamps).contains(index) ? 2.5 : 1
            }
            let calibration = try calibrate(model, clock: clock)
            XCTAssertEqual(clock.calls, stampsPerMeasurement)
            XCTAssertEqual(calibration.roundCost, 2.5, accuracy: 1e-9)
            XCTAssertEqual(calibration.plainStepSeconds, 1, accuracy: 1e-9)
            XCTAssertEqual(calibration.measurements, 1)
            XCTAssertEqual(calibration.outcome, .passed)
            XCTAssertEqual(calibration.fallback, DFlash2Fallback(roundCost: 2.5, blockSize: 5))
        }
    }

    /// A noisy measurement is repeated once: noisy twice declines without a
    /// self-check, noisy once then steady passes. Breaks if the retry is
    /// skipped, repeated, or its result ignored.
    func testCalibrationRetriesANoisyMeasurementOnce() throws {
        try onCPU {
            let model = try tinyTarget()
            let noisy = ScriptedClock { $0 % 2 == 0 ? 1 : 3 }
            let declined = try calibrate(model, clock: noisy)
            XCTAssertEqual(declined.outcome, .noisy)
            XCTAssertEqual(declined.measurements, 2)
            XCTAssertEqual(noisy.calls, 2 * stampsPerMeasurement)
            XCTAssertGreaterThan(declined.plainSpread, DFlash2Calibration.noiseThreshold)

            let first = stampsPerMeasurement
            let settles = ScriptedClock { $0 < first ? ($0 % 2 == 0 ? 1 : 3) : 1 }
            let passed = try calibrate(model, clock: settles)
            XCTAssertEqual(passed.outcome, .passed)
            XCTAssertEqual(passed.measurements, 2)
            XCTAssertEqual(passed.plainSpread, 0)
        }
    }

    /// Plain steps that run 10% slower after the rounds than before them make
    /// the measurement noisy, though each phase is steady on its own; so do
    /// spread round intervals. Breaks if the level-shift or the round-spread
    /// rule is removed.
    func testCalibrationDeclinesALevelShiftAndSpreadRounds() throws {
        try onCPU {
            let model = try tinyTarget()
            let (plainStamps, roundStamps) = (self.plainStamps, self.roundStamps)
            let shifted = ScriptedClock { call in
                let index = call % (2 * plainStamps + roundStamps)
                if index < plainStamps { return 1 }
                return index < plainStamps + roundStamps ? 2 : 1.1
            }
            let declined = try calibrate(model, clock: shifted)
            XCTAssertEqual(declined.outcome, .noisy)
            XCTAssertEqual(declined.measurements, 2)
            XCTAssertEqual(declined.plainSpread, 0)
            XCTAssertEqual(declined.plainDrift, 0.1 / 1.05, accuracy: 1e-9)

            let spreadRounds = ScriptedClock { call in
                let index = call % (2 * plainStamps + roundStamps)
                let inRounds = (plainStamps ..< plainStamps + roundStamps).contains(index)
                return inRounds ? (index % 2 == 0 ? 1.5 : 2.5) : 1
            }
            let spread = try calibrate(model, clock: spreadRounds)
            XCTAssertEqual(spread.outcome, .noisy)
            XCTAssertEqual(spread.plainDrift, 0)
            XCTAssertGreaterThan(spread.roundSpread, DFlash2Calibration.roundSpreadThreshold)

            // A 3% shift is within the threshold and passes.
            let small = ScriptedClock { call in
                let index = call % (2 * plainStamps + roundStamps)
                if index < plainStamps { return 1 }
                return index < plainStamps + roundStamps ? 2 : 1.03
            }
            XCTAssertEqual(try calibrate(model, clock: small).outcome, .passed)
        }
    }

    /// Round intervals of zero give a round cost of zero and a spread that is
    /// not a number; that is a noisy measurement, not a pass at a cost no loss
    /// would ever trip. Breaks if the noise rule lets a NaN through.
    func testCalibrationDeclinesZeroIntervals() throws {
        try onCPU {
            let model = try tinyTarget()
            let (plainStamps, roundStamps) = (self.plainStamps, self.roundStamps)
            let frozen = ScriptedClock { call in
                let index = call % (2 * plainStamps + roundStamps)
                return (plainStamps ..< plainStamps + roundStamps).contains(index) ? 0 : 1
            }
            XCTAssertEqual(try calibrate(model, clock: frozen).outcome, .noisy)
        }
    }

    /// A round costing the block width declines without a self-check.
    /// Breaks if the cost guard is dropped or compares with `>`.
    func testCalibrationDeclinesARoundThatCannotPay() throws {
        try onCPU {
            let model = try tinyTarget()
            let (plainStamps, roundStamps) = (self.plainStamps, self.roundStamps)
            let clock = ScriptedClock {
                (plainStamps ..< plainStamps + roundStamps).contains($0) ? 5 : 1
            }
            XCTAssertEqual(try calibrate(model, clock: clock).outcome, .cannotPay)
        }
    }

    private func selfCheck(
        _ model: Qwen35, blockSize: Int = 5, faults: DFlash2SelfCheckFaults = .init()
    ) throws -> DFlash2SelfCheckFailure? {
        let prefix = try DFlash2CalibrationPrefix(
            target: model, vocabularySize: vocabulary, parameters: sampling(maxTokens: 4))
        // Float32 on the CPU: the verify pass and the steps differ by rounding only.
        return try dflash2SelfCheck(
            target: model, prefix: prefix, blockSize: blockSize, captureLayers: captureLayers,
            nearTieGap: 1e-3, faults: faults)
    }

    /// The verify pass and a partial commit agree with plain steps at every
    /// width. Breaks if the self-check feeds the verify pass other tokens or
    /// positions than the plain chain, or commits differently from a round.
    func testSelfCheckPassesACorrectTarget() throws {
        try onCPU {
            let model = try tinyTarget()
            for width in [2, 5, 8] {
                XCTAssertNil(try selfCheck(model, blockSize: width), "width \(width)")
            }
        }
    }

    /// Corrupted verify logits fail the self-check. Breaks if the check skips
    /// finiteness or the argmax comparison.
    func testSelfCheckFailsACorruptedVerifyPass() throws {
        try onCPU {
            let model = try tinyTarget()
            let poisoned = try selfCheck(
                model,
                faults: .init(verifyLogits: { logits in
                    MLX.where(
                        MLXArray(Int32(0) ..< Int32(logits.dim(0))).expandedDimensions(axis: 1)
                            .== 3, MLXArray(Float.nan), logits)
                }))
            XCTAssertEqual(poisoned, .nonFiniteVerifyRow(row: 3))
            let negated = try selfCheck(model, faults: .init(verifyLogits: { -$0 }))
            guard case .verifyDisagrees(row: 0, _, _, _) = negated else {
                return XCTFail("negated logits: \(String(describing: negated))")
            }
        }
    }

    /// Committing one row too many or too few fails the self-check. Breaks if
    /// the self-check compares the step after the commit with the wrong plain
    /// row, or does not commit what a round commits.
    func testSelfCheckFailsACommitOffByOne() throws {
        try onCPU {
            let model = try tinyTarget()
            for skew in [1, -1] {
                let failure = try selfCheck(model, faults: .init(commitSkew: skew))
                guard case .commitDisagrees = failure else {
                    return XCTFail("skew \(skew): \(String(describing: failure))")
                }
            }
        }
    }

    /// Calibrating over the compiled verify and drafter traces leaves nothing
    /// holding the target or the drafter. Breaks if the calibration keeps a
    /// cache, an iterator or a trace that references them.
    func testCalibrationReleasesTargetAndDrafter() throws {
        weak var weakTarget: Qwen35?
        weak var weakDrafter: DFlash2DraftModel?
        weak var weakLayerWeight: MLXArray?
        try autoreleasepool {
            let target = try tinyTarget(dtype: .bfloat16, linearHeadDim: 32)
            let drafter = tinyDrafter()
            drafter.update(parameters: drafter.parameters().mapValues { $0.asType(.bfloat16) })
            eval(drafter)
            weakTarget = target
            weakDrafter = drafter
            weakLayerWeight = target.languageModel.model.layers[2].linearAttn!.outProj.weight
            let calibration = try dflash2Calibrate(
                target: target, drafter: drafter, blockSize: 5,
                parameters: GenerateParameters(temperature: 0.6, topP: 0.8))
            XCTAssertGreaterThan(calibration.roundSeconds, 0)
            XCTAssertGreaterThan(target.languageModel.model.dflash2VerifyTraces.traces.count, 0)
            Stream.gpu.synchronize()
        }
        XCTAssertNil(weakTarget, "the target model outlives its last reference")
        XCTAssertNil(weakDrafter, "the drafter outlives its last reference")
        XCTAssertNil(weakLayerWeight, "a target weight array outlives the target")
    }
}

// MARK: - Clock hold

final class GPUClockHoldTests: XCTestCase {

    /// Start and stop are idempotent and a hold restarts after a stop. Breaks
    /// if a repeated start or stop deadlocks or leaves the state wrong.
    func testStartAndStopAreIdempotent() throws {
        let hold = try XCTUnwrap(GPUClockHold())
        hold.stop()
        hold.start()
        hold.start()
        XCTAssertTrue(hold.isHolding)
        Thread.sleep(forTimeInterval: 0.05)
        hold.stop()
        hold.stop()
        XCTAssertFalse(hold.isHolding)
        hold.start()
        hold.stop()
        XCTAssertFalse(hold.isHolding)
    }

    /// Dropping a running hold stops it. Breaks if the spinner retains the hold.
    func testDroppingARunningHoldStopsIt() throws {
        weak var weakHold: GPUClockHold?
        try autoreleasepool {
            let hold = try XCTUnwrap(GPUClockHold())
            weakHold = hold
            hold.start()
            Thread.sleep(forTimeInterval: 0.02)
        }
        XCTAssertNil(weakHold, "a running hold outlives its last reference")
    }
}
