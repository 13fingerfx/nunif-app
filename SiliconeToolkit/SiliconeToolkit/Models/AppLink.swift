import SwiftUI

struct AppLink: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String
    let urlString: String
    let systemImage: String
    let tint: Color

    var url: URL? { URL(string: urlString) }
}

extension AppLink {
    // Edit titles, subtitles, and URLs here to update the Links section
    static let all: [AppLink] = [
        AppLink(
            title: "Website",
            subtitle: "13fingerfx.com",              // TODO: confirm URL
            urlString: "https://13fingerfx.com",
            systemImage: "globe",
            tint: Brand.orange
        ),
        AppLink(
            title: "Shop",
            subtitle: "Products & kits",
            urlString: "https://13fingerfx.com/shop", // TODO: confirm URL
            systemImage: "bag.fill",
            tint: Brand.orange
        ),
        AppLink(
            title: "Instagram",
            subtitle: "@13fingerfx",                  // TODO: confirm handle
            urlString: "https://instagram.com/13fingerfx",
            systemImage: "camera.fill",
            tint: Color(red: 0.83, green: 0.12, blue: 0.56)
        ),
    ]
}
