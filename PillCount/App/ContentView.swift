//
//  ContentView.swift
//  PillCount
//
//  Full-screen live preview with per-pill outlines, the count HUD, and the
//  control bar (freeze, manual +/- override, torch, calibration).
//

import SwiftUI

struct ContentView: View {
    @StateObject private var model = PillCountModel()
    @State private var showCalibration = false
    @State private var showResetConfirmation = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if model.camera.authorization == .denied {
                cameraDeniedView
            } else {
                ZStack {
                    if let frozen = model.frozen {
                        // Frozen still, displayed with the same aspect-fill
                        // as the live preview so the overlay stays aligned.
                        Image(uiImage: frozen.image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        CameraPreviewView(session: model.camera.session) {
                            model.camera.focus(atDevicePoint: $0)
                        }
                    }
                    DetectionOverlayView(
                        frame: model.overlayFrame,
                        locked: model.countState.isLocked || model.isFrozen)
                }
                .ignoresSafeArea()

                VStack(spacing: 10) {
                    CountHUDView(count: model.displayedCount,
                                 state: model.countState,
                                 isFrozen: model.isFrozen,
                                 manualAdjustment: model.manualAdjustment)
                        .padding(.top, 8)
                    if !model.batches.isEmpty {
                        BottleTotalView(
                            total: model.totalCount,
                            trays: model.batches.count,
                            onUndo: { model.undoLastBatch() },
                            onReset: { showResetConfirmation = true })
                    }
                    Spacer()
                    if model.isFrozen {
                        addToTotalButton
                    }
                    controlBar
                        .padding(.bottom, 12)
                }
                .confirmationDialog(
                    "Reset the bottle total of \(model.totalCount) pills?",
                    isPresented: $showResetConfirmation,
                    titleVisibility: .visible)
                {
                    Button("Reset total", role: .destructive) {
                        model.resetTotal()
                    }
                }
            }
        }
        .statusBarHidden()
        .onAppear { model.start() }
        .onDisappear { model.stop() }
        .sheet(isPresented: $showCalibration) {
            CalibrationView(model: model)
        }
    }

    // MARK: - Controls

    /// Commits the verified frozen count to the running bottle total and
    /// resumes live counting for the next tray. Only reachable while frozen,
    /// so every tray in the total was inspectable before being added.
    private var addToTotalButton: some View {
        Button(action: { model.addFrozenCountToTotal() }) {
            Label("Add \(model.displayedCount ?? 0) to total",
                  systemImage: "plus.circle.fill")
                .font(.title3.weight(.bold))
                .foregroundColor(.white)
                .padding(.horizontal, 24)
                .padding(.vertical, 13)
                .background(.green, in: Capsule())
                .shadow(color: .black.opacity(0.4), radius: 5)
        }
        .accessibilityHint("Adds this tray's count to the bottle total and "
                           + "resumes counting")
    }

    private var controlBar: some View {
        HStack(spacing: 20) {
            // Manual override: pharmacist corrects an edge case on the
            // frozen frame. Only enabled while frozen so the correction
            // applies to a specific, inspectable image.
            roundButton("minus", disabled: !model.isFrozen) {
                model.adjustCount(by: -1)
            }

            Button(action: { model.toggleFreeze() }) {
                ZStack {
                    Circle()
                        .fill(model.isFrozen ? Color.blue : Color.white)
                        .frame(width: 76, height: 76)
                    Image(systemName: model.isFrozen
                          ? "play.fill" : "pause.fill")
                        .font(.system(size: 30, weight: .bold))
                        .foregroundColor(model.isFrozen ? .white : .black)
                }
            }
            .accessibilityLabel(model.isFrozen ? "Resume live counting"
                                               : "Freeze frame")

            roundButton("plus", disabled: !model.isFrozen) {
                model.adjustCount(by: 1)
            }
        }
        .overlay(alignment: .leading) {
            roundButton(model.camera.isTorchOn
                        ? "flashlight.on.fill" : "flashlight.off.fill") {
                model.camera.toggleTorch()
            }
            .padding(.leading, 28)
        }
        .overlay(alignment: .trailing) {
            roundButton("checklist") {
                showCalibration = true
            }
            .padding(.trailing, 28)
        }
        .frame(maxWidth: .infinity)
    }

    private func roundButton(_ systemImage: String, disabled: Bool = false,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.white)
                .frame(width: 52, height: 52)
                .background(.black.opacity(0.55), in: Circle())
                .overlay(Circle().strokeBorder(.white.opacity(0.35)))
        }
        .disabled(disabled)
        .opacity(disabled ? 0.35 : 1)
    }

    private var cameraDeniedView: some View {
        VStack(spacing: 14) {
            Image(systemName: "video.slash")
                .font(.system(size: 44))
            Text("Camera access is required to count pills.")
                .multilineTextAlignment(.center)
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .foregroundColor(.white)
        .padding()
    }
}
