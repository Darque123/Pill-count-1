//
//  PillCountModel.swift
//  PillCount
//
//  Central state machine: wires camera frames into the detection engine,
//  smooths raw counts, and manages freeze-frame, manual override, and the
//  accuracy log. All published state is main-thread.
//

import Combine
import Foundation
import SwiftUI

@MainActor
final class PillCountModel: ObservableObject {
    /// A frozen capture: still image + the detection frame it was made from.
    struct FrozenCapture {
        let image: UIImage
        let frame: DetectionFrame
    }

    let camera = CameraManager()
    let log = CountLog()

    /// Latest processed frame — drives the pill outlines.
    @Published private(set) var frame: DetectionFrame?
    /// Smoothed user-facing count state (counting… / LOCKED).
    @Published private(set) var countState: CountState = .searching
    /// Non-nil while the freeze button is engaged.
    @Published private(set) var frozen: FrozenCapture?
    /// Manual +/- correction applied on top of a frozen count.
    @Published private(set) var manualAdjustment = 0

    private let engine = DetectionEngine()
    private var smoother = CountSmoother()
    private var cancellables: Set<AnyCancellable> = []

    init() {
        // Nested ObservableObjects don't propagate changes automatically;
        // forward them so views observing this model re-render on camera
        // authorization / torch / log changes.
        camera.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
        log.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)

        engine.onFrame = { [weak self] frame in
            guard let self, self.frozen == nil else { return }
            self.frame = frame
            self.countState = self.smoother.ingest(frame.count)
        }
        camera.onFrame = { [weak self] buffer in
            self?.engine.process(buffer)
        }
    }

    func start() { camera.start() }
    func stop() { camera.stop() }

    var isFrozen: Bool { frozen != nil }

    /// The number shown in the big HUD digit.
    var displayedCount: Int? {
        if let frozen { return frozen.frame.count + manualAdjustment }
        return countState.displayCount
    }

    /// The frame whose contours the overlay should draw.
    var overlayFrame: DetectionFrame? { frozen?.frame ?? frame }

    // MARK: - Freeze / capture

    func toggleFreeze() {
        if frozen != nil {
            unfreeze()
        } else if let capture = engine.captureLastFrame() {
            engine.setPaused(true)
            manualAdjustment = 0
            frozen = FrozenCapture(image: capture.image, frame: capture.frame)
        }
    }

    private func unfreeze() {
        frozen = nil
        manualAdjustment = 0
        smoother.reset()
        countState = .searching
        engine.setPaused(false)
    }

    // MARK: - Manual override

    /// Pharmacist correction of an edge case. Every use is logged so
    /// systematic detector errors are visible in the accuracy record.
    func adjustCount(by delta: Int) {
        guard let frozen else { return }
        let machineCount = frozen.frame.count
        manualAdjustment = max(-machineCount, manualAdjustment + delta)
        log.recordOverride(machineCount: machineCount,
                           adjustment: manualAdjustment)
    }

    // MARK: - Calibration / accuracy testing

    /// Record the true count for the current frozen frame; the app logs its
    /// own error so accuracy can be measured over time and across pill types.
    func recordCalibration(trueCount: Int, note: String) {
        guard let frozen else { return }
        log.recordCalibration(machineCount: frozen.frame.count,
                              adjustment: manualAdjustment,
                              trueCount: trueCount,
                              note: note)
    }
}
