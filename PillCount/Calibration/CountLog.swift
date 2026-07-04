//
//  CountLog.swift
//  PillCount
//
//  Persistent, on-device record of manual overrides and calibration checks.
//  Overrides reveal where the detector needed human correction; calibration
//  entries measure counting error against known-true counts over time.
//  Stored as JSON in the app's Documents directory — nothing leaves the
//  device unless the user explicitly shares the CSV export.
//

import Foundation

struct CountLogEntry: Codable, Identifiable {
    enum Kind: String, Codable {
        case override      // pharmacist used +/- to correct a count
        case calibration   // user entered the known-true count for a frame
    }

    var id = UUID()
    var date = Date()
    var kind: Kind
    /// What the detector counted, before any human correction.
    var machineCount: Int
    /// Manual +/- adjustment in effect when the entry was recorded.
    var adjustment: Int
    /// Ground truth entered by the user (calibration entries only).
    var trueCount: Int?
    var note: String?

    /// machineCount - trueCount (calibration entries only): positive means
    /// the detector overcounted.
    var error: Int? {
        guard let trueCount else { return nil }
        return machineCount - trueCount
    }
}

@MainActor
final class CountLog: ObservableObject {
    @Published private(set) var entries: [CountLogEntry] = []

    private let fileURL: URL = {
        let docs = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("count-log.json")
    }()

    init() {
        load()
    }

    func recordOverride(machineCount: Int, adjustment: Int) {
        // Collapse repeated taps on +/- for the same frozen frame into one
        // entry so the log reflects corrections, not button presses.
        if let last = entries.last, last.kind == .override,
           last.machineCount == machineCount,
           Date().timeIntervalSince(last.date) < 30 {
            entries[entries.count - 1].adjustment = adjustment
            entries[entries.count - 1].date = Date()
        } else {
            entries.append(CountLogEntry(kind: .override,
                                         machineCount: machineCount,
                                         adjustment: adjustment))
        }
        save()
    }

    func recordCalibration(machineCount: Int, adjustment: Int,
                           trueCount: Int, note: String) {
        entries.append(CountLogEntry(kind: .calibration,
                                     machineCount: machineCount,
                                     adjustment: adjustment,
                                     trueCount: trueCount,
                                     note: note.isEmpty ? nil : note))
        save()
    }

    func clear() {
        entries.removeAll()
        save()
    }

    // MARK: - Accuracy statistics (calibration entries)

    struct Stats {
        let samples: Int
        let exactRate: Double     // fraction with zero error
        let meanAbsError: Double
        let bias: Double          // mean signed error; + = overcounting
    }

    var stats: Stats? {
        let errors = entries.compactMap { $0.error }
        guard !errors.isEmpty else { return nil }
        let exact = errors.filter { $0 == 0 }.count
        return Stats(
            samples: errors.count,
            exactRate: Double(exact) / Double(errors.count),
            meanAbsError: Double(errors.map { abs($0) }.reduce(0, +))
                / Double(errors.count),
            bias: Double(errors.reduce(0, +)) / Double(errors.count))
    }

    // MARK: - Export

    /// CSV of all entries, written to a temporary file for the share sheet.
    func exportCSV() -> URL? {
        var csv = "date,kind,machine_count,adjustment,true_count,error,note\n"
        let formatter = ISO8601DateFormatter()
        for entry in entries {
            let fields = [
                formatter.string(from: entry.date),
                entry.kind.rawValue,
                String(entry.machineCount),
                String(entry.adjustment),
                entry.trueCount.map(String.init) ?? "",
                entry.error.map(String.init) ?? "",
                (entry.note ?? "").replacingOccurrences(of: ",", with: ";"),
            ]
            csv += fields.joined(separator: ",") + "\n"
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pillcount-log.csv")
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            return nil
        }
    }

    // MARK: - Persistence

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode(
                  [CountLogEntry].self, from: data)
        else { return }
        entries = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(entries) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
