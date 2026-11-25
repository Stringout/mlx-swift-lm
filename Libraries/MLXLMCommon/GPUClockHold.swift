// Copyright © 2026 Apple Inc.

// A GPU clock hold: one SIMD group kept busy on its own command queue, so the
// GPU does not clock down between the short bursts of one-token decode.

import Foundation
import Metal

/// Keeps one SIMD group of the GPU spinning, in short command buffers on a
/// queue of its own, from ``start()`` to ``stop()``.
///
/// It computes nothing anyone reads, so it cannot change any output; it costs
/// one threadgroup's share of one GPU core while held. ``start()`` and
/// ``stop()`` are idempotent, and once ``stop()`` returns no buffer of the hold
/// is queued or running.
public final class GPUClockHold: @unchecked Sendable {
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let sink: MTLBuffer
    private let iterations: UInt32
    private let lock = NSLock()
    private var run: Run?

    /// One start-to-stop span: its worker checks `stopping` and signals `finished`.
    private final class Run: @unchecked Sendable {
        let lock = NSLock()
        var stopping = false
        let finished = DispatchSemaphore(value: 0)

        var shouldStop: Bool { lock.withLock { stopping } }
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void clock_hold(
            device float *sink [[buffer(0)]], constant uint &iterations [[buffer(1)]],
            uint tid [[thread_index_in_threadgroup]]) {
            float x = float(tid);
            for (uint i = 0; i < iterations; ++i) { x = fma(x, 0.99990f, 0.00013f); }
            if (x == -1.0f) { sink[tid] = x; }
        }
        """

    /// - Parameters:
    ///   - device: the GPU to hold; the system default when nil.
    ///   - iterations: dependent multiply-adds per command buffer, about
    ///     0.2 ms of one SIMD group at the default.
    /// - Returns: nil when there is no Metal device or the kernel does not build.
    public init?(device: MTLDevice? = nil, iterations: UInt32 = 1 << 16) {
        guard let device = device ?? MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let library = try? device.makeLibrary(source: Self.source, options: nil),
            let function = library.makeFunction(name: "clock_hold"),
            let pipeline = try? device.makeComputePipelineState(function: function),
            let sink = device.makeBuffer(length: 32 * 4, options: .storageModePrivate)
        else { return nil }
        queue.label = "GPUClockHold"
        self.queue = queue
        self.pipeline = pipeline
        self.sink = sink
        self.iterations = iterations
    }

    deinit {
        stop()
    }

    public var isHolding: Bool { lock.withLock { run != nil } }

    /// Start spinning; a no-op while already holding.
    public func start() {
        lock.withLock {
            guard run == nil else { return }
            let run = Run()
            self.run = run
            let thread = Thread { [queue, pipeline, sink, iterations] in
                Self.spin(
                    run: run, queue: queue, pipeline: pipeline, sink: sink,
                    iterations: iterations)
            }
            thread.name = "GPUClockHold"
            thread.qualityOfService = .userInitiated
            thread.start()
        }
    }

    /// Stop spinning and wait for the last buffer; a no-op when not holding.
    public func stop() {
        let run: Run? = lock.withLock {
            defer { self.run = nil }
            return self.run
        }
        guard let run else { return }
        run.lock.withLock { run.stopping = true }
        run.finished.wait()
    }

    /// Two buffers in flight, so the GPU never waits on the host between them.
    private static func spin(
        run: Run, queue: MTLCommandQueue, pipeline: MTLComputePipelineState, sink: MTLBuffer,
        iterations: UInt32
    ) {
        let slots = DispatchSemaphore(value: 2)
        var last: MTLCommandBuffer?
        var iterations = iterations
        while !run.shouldStop {
            slots.wait()
            guard let buffer = queue.makeCommandBuffer(),
                let encoder = buffer.makeComputeCommandEncoder()
            else {
                slots.signal()
                break
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(sink, offset: 0, index: 0)
            encoder.setBytes(&iterations, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
            buffer.addCompletedHandler { _ in slots.signal() }
            buffer.commit()
            last = buffer
        }
        last?.waitUntilCompleted()
        run.finished.signal()
    }
}
