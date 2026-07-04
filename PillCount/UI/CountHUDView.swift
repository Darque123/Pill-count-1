//
//  CountHUDView.swift
//  PillCount
//
//  The big count digit and the trust indicator. The state badge is explicit
//  about when the number can be relied on: green "LOCKED ✓" only after the
//  count has been stable across consecutive frames.
//

import SwiftUI

struct CountHUDView: View {
    let count: Int?
    let state: CountState
    let isFrozen: Bool
    let manualAdjustment: Int
    /// Non-nil when the ML detector and the classical cross-check disagree
    /// (ML count minus classical count) — the pharmacist should verify.
    var crossCheckDelta: Int? = nil

    var body: some View {
        VStack(spacing: 6) {
            Text(count.map(String.init) ?? "–")
                .font(.system(size: 88, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundColor(.white)
                .shadow(color: .black.opacity(0.8), radius: 4)
                .contentTransition(.numericText())
                .animation(.easeOut(duration: 0.15), value: count)

            badge
                .font(.headline.weight(.semibold))
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(badgeColor.opacity(0.9), in: Capsule())
                .foregroundColor(.white)

            if manualAdjustment != 0 {
                Text("manually adjusted \(manualAdjustment > 0 ? "+" : "")\(manualAdjustment)")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.purple.opacity(0.9), in: Capsule())
                    .foregroundColor(.white)
            }

            // The two detectors disagree: don't hide it — this is exactly
            // the frame the pharmacist should freeze and verify by eye.
            if let delta = crossCheckDelta {
                Label("cross-check differs (\(delta > 0 ? "+" : "")\(delta)) — verify",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(.red.opacity(0.9), in: Capsule())
                    .foregroundColor(.white)
            }
        }
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var badge: some View {
        if isFrozen {
            Label("FROZEN", systemImage: "snowflake")
        } else {
            switch state {
            case .searching:
                Text("searching…")
            case .counting:
                Text("counting…")
            case .locked:
                Label("LOCKED", systemImage: "checkmark")
            }
        }
    }

    private var badgeColor: Color {
        if isFrozen { return .blue }
        switch state {
        case .searching: return .gray
        case .counting: return .orange
        case .locked: return .green
        }
    }
}

/// Running multi-tray total for the current bottle, with undo/reset.
/// Shown whenever at least one tray has been committed.
struct BottleTotalView: View {
    let total: Int
    let trays: Int
    let onUndo: () -> Void
    let onReset: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 0) {
                Text("BOTTLE TOTAL")
                    .font(.caption2.weight(.bold))
                    .foregroundColor(.white.opacity(0.7))
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text("\(total)")
                        .font(.system(size: 34, weight: .bold,
                                      design: .rounded))
                        .monospacedDigit()
                        .foregroundColor(.white)
                    Text("\(trays) \(trays == 1 ? "tray" : "trays")")
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.7))
                }
            }

            Spacer(minLength: 0)

            Button(action: onUndo) {
                Image(systemName: "arrow.uturn.backward")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 38, height: 38)
                    .background(.white.opacity(0.15), in: Circle())
            }
            .accessibilityLabel("Remove last tray from total")

            Button(action: onReset) {
                Image(systemName: "trash")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 38, height: 38)
                    .background(.white.opacity(0.15), in: Circle())
            }
            .accessibilityLabel("Reset bottle total")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.black.opacity(0.6),
                    in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16)
            .strokeBorder(.white.opacity(0.25)))
        .padding(.horizontal, 24)
    }
}
