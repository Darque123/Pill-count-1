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
