import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Calculators") {
                    NavigationLink {
                        DeadenerCalculatorView()
                    } label: {
                        Label("Deadener Calculator", systemImage: "flask.fill")
                    }
                }
            }
            .navigationTitle("13 Finger FX")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

#Preview {
    ContentView()
}
