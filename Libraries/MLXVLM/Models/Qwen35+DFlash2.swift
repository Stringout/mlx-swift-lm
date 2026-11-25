// Copyright © 2026 Apple Inc.

// The Qwen 3.5 VLM language model as a DFlash2 target. The verify pass follows
// the MLXLLM Qwen 3.5 target of ml-explore/mlx-swift-lm#607 (spokvulcan), on
// this model's compiled decode segments and M-RoPE.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import ObjectiveC

/// Mirrors `ropeDeltasKey` in Qwen35.swift.
private let ropeDeltasKey = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

/// Mirrors `supportsExactCompiledDecode` in Qwen35.swift.
private func supportsExactCompiledVerify(_ dtype: DType) -> Bool {
    dtype == .float16 || dtype == .bfloat16
}

extension Qwen35Language.GatedDeltaNet {
    /// `forward` over a verify block: the scan takes the masked route replay
    /// takes, and the recurrent state comes back as a capture. Writes no cache.
    func verifyForward(
        _ x: MLXArray, convState: MLXArray, recState: MLXArray
    ) -> (output: MLXArray, capture: GatedDeltaCapture) {
        let B = x.dim(0)
        let S = x.dim(1)
        let (qkv, z, b, a) = projectInputs(x, batch: B, sequence: S)
        let convInput = concatenated([convState, qkv], axis: 1)
        let convOut = silu(conv1d(convInput))
        let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = split[0].reshaped(B, S, numKHeads, headKDim)
        let k = split[1].reshaped(B, S, numKHeads, headKDim)
        let v = split[2].reshaped(B, S, numVHeads, headVDim)

        // Mirrors the q/k normalization in `GatedDeltaNet.forward`.
        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qkEps = 1e-6 / Float(headKDim)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: qkEps)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: qkEps)

        let (out, _) = gatedDeltaVerifyUpdate(
            q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: recState, useKernel: !training)
        let capture = GatedDeltaCapture(
            convInput: convInput, q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog,
            dtBias: dtBias, initialState: recState)
        return (outProj(norm(out, gate: z).reshaped(B, S, -1)), capture)
    }
}

extension Qwen35Language.DecoderLayer {
    /// `linearLayerBody` for a verify block, returning a capture instead of state.
    func linearLayerVerifyBody(
        x: MLXArray, convState: MLXArray, recState: MLXArray
    ) -> (output: MLXArray, capture: GatedDeltaCapture) {
        let (r, capture) = linearAttn!.verifyForward(
            inputLayerNorm(x), convState: convState, recState: recState)
        let h = x + r
        return (h + verifyMLP(postAttentionLayerNorm(h)), capture)
    }

    /// Mirrors `mlpForward(_:allowCompiledDecode: false)`.
    private func verifyMLP(_ x: MLXArray) -> MLXArray {
        if let moe = mlp as? Qwen35Language.SparseMoeBlock,
            type(of: moe) == Qwen35Language.SparseMoeBlock.self
        {
            return moe.forward(x)
        }
        return (mlp as! UnaryLayer)(x)
    }

    /// `attentionCacheStep` for a verify block at `position`, a `[1]` array that
    /// may be lazy: rotate, write the block's rows without moving the offset,
    /// and attend through `mask`.
    func attentionVerifyStep(
        queries: MLXArray, keys: MLXArray, values: MLXArray, cache: KVCacheSimple,
        position: MLXArray, visibleLength: Int, mask: MLXArray
    ) -> MLXArray {
        let attention = selfAttn!
        let length = queries.dim(2)
        // Text-only M-RoPE: every axis carries the same 1-D position.
        let positions = (position.asType(.int32) + MLXArray(Int32(0) ..< Int32(length)))
            .reshaped(1, 1, length)
        let (cosValues, sinValues) = attention.rotaryEmbedding(
            x: values, positionIds: broadcast(positions, to: [3, 1, length]))
        let (rotatedQueries, rotatedKeys) = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: queries, k: keys, cos: cosValues, sin: sinValues)
        let (allKeys, allValues) = cache.writeRows(
            keys: rotatedKeys, values: values, position: position, visibleLength: visibleLength)
        return MLXFast.scaledDotProductAttention(
            queries: rotatedQueries, keys: allKeys, values: allValues, scale: attention.scale,
            mask: .array(mask))
    }
}

// MARK: - Verify traces

/// Compiled verify segments for one model, one per `(segment, S, captureLayers)`.
///
/// Stored beside the model, so they are released with it. Fused input
/// projections reach them through the registration the decode traces use;
/// ``refresh(for:)`` drops them when the module tree changed since they were
/// built, which `Module.invalidateCompiledTraces()` cannot see.
final class Qwen35VerifyTraces {
    struct Key: Hashable {
        var segment: Int
        var length: Int
        var captureLayers: [Int]
    }

    let schedule: [CompiledDecodeSegment]
    let traces = KeyedCompiledTraces<Key, Qwen35Language.Model>()
    private let lock = NSLock()
    private var modules: [ObjectIdentifier] = []

    /// Test hook: run the segment bodies eagerly.
    var compiledEnabled = true

    init(_ model: Qwen35Language.Model) {
        schedule = CompiledDecodeSegment.schedule(linearLayers: model.layers.map(\.isLinear))
    }

    /// Drop every trace when `model`'s modules are not the ones they were built from.
    func refresh(for model: Qwen35Language.Model) {
        let current =
            model.leafModules().flattenedValues().map(ObjectIdentifier.init)
            + model.layers.compactMap { $0.linearAttn?.fusedProjectionTraceState.first }
            .map(ObjectIdentifier.init)
        let changed = lock.withLock {
            defer { modules = current }
            return modules != current
        }
        if changed {
            traces.invalidate()
        }
    }

    func trace(
        for model: Qwen35Language.Model, segment index: Int, length: Int, captureLayers: [Int]
    ) -> CompiledTrace<Qwen35Language.Model> {
        let segment = schedule[index]
        let isLast = index == schedule.count - 1
        let traces = self.traces
        return traces.trace(Key(segment: index, length: length, captureLayers: captureLayers)) {
            CompiledTrace<Qwen35Language.Model>(
                // Weak: the container holds this trace, so a strong capture is a cycle
                // that keeps the compiled function, and with it every weight, alive.
                state: { [weak traces] model in
                    var modules = segment.layerIndices.flatMap { layerIndex -> [Module] in
                        let layer = model.layers[layerIndex]
                        if let traces {
                            layer.linearAttn?.registerFusedProjectionTrace(traces)
                        }
                        return [layer] + (layer.linearAttn?.fusedProjectionTraceState ?? [])
                    }
                    if isLast {
                        modules.append(model.norm)
                    }
                    return modules
                },
                body: { model, args in
                    model.verifySegmentBody(
                        segment, isLast: isLast, captureLayers: captureLayers, args)
                })
        }
    }
}

nonisolated(unsafe) private var qwen35VerifyTracesKey: UInt8 = 0
private let qwen35VerifyTracesLock = NSLock()

extension Qwen35Language.Model {
    var dflash2VerifyTraces: Qwen35VerifyTraces {
        qwen35VerifyTracesLock.withLock {
            if let traces = objc_getAssociatedObject(self, &qwen35VerifyTracesKey)
                as? Qwen35VerifyTraces
            {
                return traces
            }
            let traces = Qwen35VerifyTraces(self)
            objc_setAssociatedObject(
                self, &qwen35VerifyTracesKey, traces, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return traces
        }
    }

    /// `segmentBody` for a verify block. In: as the decode segment. Out: `[x]`,
    /// six capture arrays per linear layer, the outputs of this segment's
    /// `captureLayers`, then `[queries, gate, keys, values]` when the segment
    /// opens an attention layer. The last segment's `x` is normed.
    fileprivate func verifySegmentBody(
        _ segment: CompiledDecodeSegment, isLast: Bool, captureLayers: [Int], _ args: [MLXArray]
    ) -> [MLXArray] {
        var hiddenStates = args[0]
        var captured: [MLXArray] = []

        if let post = segment.attentionPostLayer {
            hiddenStates = layers[post].attentionPostBody(
                x: hiddenStates, attention: args[1], gate: args[2])
            if captureLayers.contains(post) { captured.append(hiddenStates) }
        }

        var captures: [MLXArray] = []
        for (stateIndex, layerIndex) in segment.linearLayers.enumerated() {
            let slot = segment.stateInputOffset + 2 * stateIndex
            let result = layers[layerIndex].linearLayerVerifyBody(
                x: hiddenStates, convState: args[slot], recState: args[slot + 1])
            hiddenStates = result.output
            captures.append(contentsOf: result.capture.arrays)
            if captureLayers.contains(layerIndex) { captured.append(hiddenStates) }
        }

        if let pre = segment.attentionPreLayer {
            let (queries, gate, keys, values) = layers[pre].attentionPreBody(x: hiddenStates)
            return [hiddenStates] + captures + captured + [queries, gate, keys, values]
        }
        return [isLast ? norm(hiddenStates) : hiddenStates] + captures + captured
    }

    /// One verify pass over the decode segments. Attention rows land at the
    /// request's position without moving any offset, and recurrent state comes
    /// back as captures: nothing is committed. Returns the normed hidden state.
    func dflash2VerifyStep(
        _ request: DFlash2VerifyRequest, cache: [KVCache]
    ) -> (hidden: MLXArray, captured: [MLXArray], recurrentCaptures: [GatedDeltaCapture]) {
        let verifyTraces = dflash2VerifyTraces
        let length = request.tokens.dim(1)
        let visibleLength = request.positionUpperBound + length
        let captureLayers = request.captureLayers

        // Row `i` sees columns up to its own position, `position + i`.
        let columns = MLXArray(Int32(0) ..< Int32(visibleLength)).expandedDimensions(axis: 0)
        let rows = (request.position.asType(.int32) + MLXArray(Int32(0) ..< Int32(length)))
            .expandedDimensions(axis: 1)
        let mask = columns .< (rows + 1)

        var carry = embedTokens(request.tokens)
        let compiled =
            verifyTraces.compiledEnabled && type(of: self) == Qwen35Language.Model.self
            && !training && supportsExactCompiledVerify(carry.dtype)
            && layers.allSatisfy(\.supportsCompiledDecode)

        var pendingAttention: [MLXArray] = []
        var captured: [Int: MLXArray] = [:]
        var recurrentCaptures: [Int: GatedDeltaCapture] = [:]

        for (segmentIndex, segment) in verifyTraces.schedule.enumerated() {
            var args = [carry] + pendingAttention
            for layerIndex in segment.linearLayers {
                let mambaCache = cache[layerIndex] as! MambaCache
                args.append(mambaCache[0]!)
                args.append(mambaCache[1]!)
            }

            let isLast = segmentIndex == verifyTraces.schedule.count - 1
            let outputs =
                compiled
                ? verifyTraces.trace(
                    for: self, segment: segmentIndex, length: length, captureLayers: captureLayers
                )(self, args)
                : verifySegmentBody(segment, isLast: isLast, captureLayers: captureLayers, args)

            carry = outputs[0]
            var next = 1
            for layerIndex in segment.linearLayers {
                let gdn = layers[layerIndex].linearAttn!
                recurrentCaptures[layerIndex] = GatedDeltaCapture(
                    arrays: Array(outputs[next ..< (next + 6)]), aLog: gdn.aLog,
                    dtBias: gdn.dtBias, initialState: (cache[layerIndex] as! MambaCache)[1]!)
                next += 6
            }
            let capturedHere =
                ((segment.attentionPostLayer.map { [$0] } ?? [])
                + segment.linearLayers).filter(captureLayers.contains)
            for layerIndex in capturedHere {
                captured[layerIndex] = outputs[next]
                next += 1
            }

            pendingAttention = []
            if let pre = segment.attentionPreLayer {
                let attention = layers[pre].attentionVerifyStep(
                    queries: outputs[next], keys: outputs[next + 2], values: outputs[next + 3],
                    cache: cache[pre] as! KVCacheSimple, position: request.position,
                    visibleLength: visibleLength, mask: mask)
                pendingAttention = [attention, outputs[next + 1]]
            }
        }

        let recurrentInOrder = layers.indices.filter { layers[$0].isLinear }
            .map { recurrentCaptures[$0]! }
        return (carry, captureLayers.map { captured[$0]! }, recurrentInOrder)
    }
}

// MARK: - Target conformance

extension Qwen35: DFlash2TargetModel {
    public var dflash2Geometry: DFlash2Geometry {
        DFlash2Geometry(
            layerCount: languageModel.model.layers.count,
            hiddenSize: config.textConfiguration.hiddenSize,
            vocabularySize: config.textConfiguration.vocabularySize)
    }

    public var dflash2Embedding: Embedding { languageModel.model.embedTokens }

    public var dflash2Head: Linear? { languageModel.lmHead }

    /// Plain attention caches and gated-delta caches only, over text positions:
    /// the verify pass places block row `i` at `position + i`, so a carried
    /// M-RoPE delta must be zero.
    public func dflash2Prepare(cache: [KVCache], state: LMOutput.State?) throws {
        let model = languageModel.model
        guard cache.count == model.layers.count else {
            throw DFlash2SpeculationError.unsupportedCache(
                "\(cache.count) cache entries for \(model.layers.count) layers")
        }
        for (index, (layer, entry)) in zip(model.layers, cache).enumerated() {
            let expected: KVCache.Type = layer.isLinear ? MambaCache.self : KVCacheSimple.self
            guard type(of: entry) == expected else {
                throw DFlash2SpeculationError.unsupportedCache(
                    "layer \(index) needs \(expected), got \(type(of: entry))")
            }
        }
        if let ropeDeltas = state?[ropeDeltasKey] {
            guard (ropeDeltas .== 0).all().item(Bool.self) else {
                throw DFlash2SpeculationError.unsupportedPositions(
                    "the carried M-RoPE delta is nonzero, as after an image")
            }
        } else if cache.contains(where: { $0.offset > 0 }) {
            throw DFlash2SpeculationError.missingContinuationState(
                "a warm Qwen 3.5 cache needs the \(ropeDeltasKey.id) state its prefill returned")
        }
        model.dflash2VerifyTraces.refresh(for: model)
    }

    public func dflash2Verify(
        _ request: DFlash2VerifyRequest, cache: [KVCache]
    ) -> DFlash2VerifyResult {
        let model = languageModel.model
        let (hidden, captured, recurrentCaptures) = model.dflash2VerifyStep(request, cache: cache)
        let logits = languageModel.lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
        return DFlash2VerifyResult(
            logits: logits, hidden: captured, recurrentCaptures: recurrentCaptures)
    }
}
