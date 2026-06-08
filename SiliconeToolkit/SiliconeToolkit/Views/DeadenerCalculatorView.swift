import SwiftUI

struct DeadenerCalculatorView: View {
    @State private var totalWeightText = ""
    @State private var deadenerPercentText = ""
    @FocusState private var focusedField: Field?

    private enum Field { case totalWeight, deadenerPercent }

    private var calculation: DeadenerCalculation? {
        guard let total = Double(totalWeightText),
              let percent = Double(deadenerPercentText),
              total > 0, percent >= 0
        else { return nil }
        return DeadenerCalculation(totalWeight: total, deadenerPercent: percent)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                inputCard
                if let calc = calculation {
                    resultsCard(calc)
                }
            }
            .padding()
        }
        .navigationTitle("Deadener Calculator")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { focusedField = nil }
            }
        }
        .onTapGesture { focusedField = nil }
    }

    private var inputCard: some View {
        VStack(spacing: 0) {
            inputRow(
                label: "Total Batch Weight",
                unit: "g",
                placeholder: "e.g. 40",
                text: $totalWeightText,
                field: .totalWeight
            )
            Divider().padding(.leading, 16)
            inputRow(
                label: "Deadener %",
                unit: "%",
                placeholder: "e.g. 100",
                text: $deadenerPercentText,
                field: .deadenerPercent
            )
        }
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func inputRow(
        label: String,
        unit: String,
        placeholder: String,
        text: Binding<String>,
        field: Field
    ) -> some View {
        HStack(spacing: 12) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            TextField(placeholder, text: text)
                .keyboardType(.decimalPad)
                .multilineTextAlignment(.trailing)
                .focused($focusedField, equals: field)
                .frame(width: 90)
            Text(unit)
                .foregroundStyle(.secondary)
                .frame(width: 20, alignment: .leading)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    private func resultsCard(_ calc: DeadenerCalculation) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Mix Breakdown")
                .font(.headline)

            VStack(spacing: 10) {
                resultRow(label: "Part A", grams: calc.partA, color: .blue)
                resultRow(label: "Part B", grams: calc.partB, color: .blue)
                resultRow(label: "Deadener (D)", grams: calc.partD, color: .orange)
            }

            Divider()

            HStack {
                Text("Total")
                    .fontWeight(.semibold)
                Spacer()
                Text("\(calc.totalWeight.gramsString()) g")
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.semibold)
            }

            Text("D is \(calc.deadenerPercent.gramsString())% of combined A+B (\(calc.abCombined.gramsString()) g)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func resultRow(label: String, grams: Double, color: Color) -> some View {
        HStack(spacing: 10) {
            RoundedRectangle(cornerRadius: 3)
                .fill(color)
                .frame(width: 4, height: 22)
            Text(label)
                .foregroundStyle(.primary)
            Spacer()
            Text("\(grams.gramsString()) g")
                .font(.system(.body, design: .monospaced))
                .fontWeight(.semibold)
        }
    }
}

#Preview {
    NavigationStack {
        DeadenerCalculatorView()
    }
}
