import Foundation

struct DeadenerCalculation {
    var totalWeight: Double
    var deadenerPercent: Double

    var abCombined: Double {
        totalWeight / (1.0 + deadenerPercent / 100.0)
    }

    var partA: Double { abCombined / 2.0 }
    var partB: Double { abCombined / 2.0 }
    var partD: Double { totalWeight - abCombined }

    var isValid: Bool {
        totalWeight > 0 && deadenerPercent >= 0
    }
}

extension Double {
    func gramsString() -> String {
        String(format: "%.2f", self)
    }
}
