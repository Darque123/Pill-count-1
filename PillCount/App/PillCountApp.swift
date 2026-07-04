//
//  PillCountApp.swift
//  PillCount
//
//  Real-time, on-device pill counting for pharmacy trays. All processing is
//  local (OpenCV, classical CV) — no network access is used or required.
//

import SwiftUI

@main
struct PillCountApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
