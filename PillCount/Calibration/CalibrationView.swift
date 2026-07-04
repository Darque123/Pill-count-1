//
//  CalibrationView.swift
//  PillCount
//
//  Accuracy-test mode: with a frame frozen, the user enters the known-true
//  count and the app logs its own error. The stats section aggregates those
//  checks (exact-match rate, mean absolute error, bias) so accuracy can be
//  measured over time and across pill types before the tool is relied on.
//

import SwiftUI

struct CalibrationView: View {
    @ObservedObject var model: PillCountModel
    @Environment(\.dismiss) private var dismiss

    @State private var trueCountText = ""
    @State private var note = ""
    @State private var savedBanner = false

    var body: some View {
        NavigationStack {
            Form {
                if let frozen = model.frozen {
                    Section("Record a check for the frozen frame") {
                        LabeledContent("App counted",
                                       value: "\(frozen.frame.count)")
                        TextField("True count (count by hand)",
                                  text: $trueCountText)
                            .keyboardType(.numberPad)
                        TextField("Note (pill type, tray, lighting…)",
                                  text: $note)
                        Button("Save check") { saveCheck() }
                            .disabled(Int(trueCountText) == nil)
                        if savedBanner {
                            Label("Saved", systemImage: "checkmark.circle.fill")
                                .foregroundColor(.green)
                        }
                    }
                } else {
                    Section {
                        Text("Freeze a frame first, then enter the true count "
                             + "here to record an accuracy check.")
                            .foregroundColor(.secondary)
                    }
                }

                if let stats = model.log.stats {
                    Section("Accuracy so far") {
                        LabeledContent("Checks", value: "\(stats.samples)")
                        LabeledContent("Exact-count rate",
                                       value: String(format: "%.1f%%",
                                                     stats.exactRate * 100))
                        LabeledContent("Mean absolute error",
                                       value: String(format: "%.2f pills",
                                                     stats.meanAbsError))
                        LabeledContent("Bias",
                                       value: String(format: "%+.2f pills",
                                                     stats.bias))
                    }
                }

                Section("Log") {
                    if model.log.entries.isEmpty {
                        Text("No entries yet.")
                            .foregroundColor(.secondary)
                    }
                    ForEach(model.log.entries.reversed()) { entry in
                        LogEntryRow(entry: entry)
                    }
                    if let url = model.log.exportCSV(),
                       !model.log.entries.isEmpty {
                        ShareLink("Export CSV", item: url)
                    }
                    if !model.log.entries.isEmpty {
                        Button("Clear log", role: .destructive) {
                            model.log.clear()
                        }
                    }
                }
            }
            .navigationTitle("Accuracy & Calibration")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func saveCheck() {
        guard let trueCount = Int(trueCountText) else { return }
        model.recordCalibration(trueCount: trueCount, note: note)
        trueCountText = ""
        note = ""
        savedBanner = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            savedBanner = false
        }
    }
}

private struct LogEntryRow: View {
    let entry: CountLogEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(entry.date, style: .date)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            switch entry.kind {
            case .calibration:
                let err = entry.error ?? 0
                Text("counted \(entry.machineCount), true "
                     + "\(entry.trueCount ?? 0) → "
                     + (err == 0 ? "exact ✓" : "error \(err > 0 ? "+" : "")\(err)"))
                    .font(.caption)
                    .foregroundColor(err == 0 ? .green : .red)
            case .override:
                Text("counted \(entry.machineCount), adjusted "
                     + "\(entry.adjustment > 0 ? "+" : "")\(entry.adjustment)")
                    .font(.caption)
                    .foregroundColor(.orange)
            case .batch:
                Text("added \(entry.machineCount + entry.adjustment) pills"
                     + (entry.adjustment != 0
                        ? " (counted \(entry.machineCount), adjusted "
                          + "\(entry.adjustment > 0 ? "+" : "")\(entry.adjustment))"
                        : ""))
                    .font(.caption)
                    .foregroundColor(.blue)
            case .batchUndo:
                Text("removed \(entry.machineCount) pills")
                    .font(.caption)
                    .foregroundColor(.orange)
            }
            if let note = entry.note {
                Text(note).font(.caption2).foregroundColor(.secondary)
            }
        }
    }

    private var title: String {
        switch entry.kind {
        case .calibration: return "Check"
        case .override: return "Override"
        case .batch: return "Tray added"
        case .batchUndo: return "Tray removed"
        }
    }
}
