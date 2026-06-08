import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    brandHeader
                    VStack(spacing: 24) {
                        calculatorsSection
                        linksSection
                    }
                    .padding(.top, 28)
                    .padding(.bottom, 48)
                }
            }
            .background(Color(.systemGroupedBackground))
            .toolbar(.hidden, for: .navigationBar)
        }
    }

    // MARK: - Brand Header

    private var brandHeader: some View {
        VStack(spacing: 10) {
            Group {
                if UIImage(named: "AppLogo") != nil {
                    Image("AppLogo")
                        .resizable()
                        .scaledToFit()
                        .frame(height: 80)
                } else {
                    Text("13FX")
                        .font(.system(size: 52, weight: .black))
                        .foregroundStyle(Brand.orange)
                }
            }
            .padding(.top, 36)

            Text(Brand.appName.uppercased())
                .font(.title3)
                .fontWeight(.black)
                .tracking(2)
                .foregroundStyle(.primary)

            Text(Brand.tagline.uppercased())
                .font(.caption2)
                .fontWeight(.bold)
                .tracking(4)
                .foregroundStyle(Brand.orange)

            Rectangle()
                .fill(Brand.orange)
                .frame(height: 2)
                .frame(maxWidth: 48)
                .padding(.top, 4)
                .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity)
        .background(Color(.systemBackground))
    }

    // MARK: - Calculators

    private var calculatorsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Calculators")
            VStack(spacing: 0) {
                NavigationLink {
                    DeadenerCalculatorView()
                } label: {
                    linkRow(
                        title: "Deadener Calculator",
                        subtitle: "Work out A, B & D weights",
                        systemImage: "flask.fill",
                        tint: Brand.orange
                    )
                }
                .buttonStyle(.plain)
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 16)
    }

    // MARK: - Links

    private var linksSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Links")
            VStack(spacing: 0) {
                ForEach(Array(AppLink.all.enumerated()), id: \.element.id) { index, link in
                    if let url = link.url {
                        Link(destination: url) {
                            linkRow(
                                title: link.title,
                                subtitle: link.subtitle,
                                systemImage: link.systemImage,
                                tint: link.tint
                            )
                        }
                        .buttonStyle(.plain)
                        if index < AppLink.all.count - 1 {
                            Divider().padding(.leading, 56)
                        }
                    }
                }
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 16)
    }

    // MARK: - Shared components

    private func sectionLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundStyle(.secondary)
            .padding(.leading, 4)
    }

    private func linkRow(title: String, subtitle: String, systemImage: String, tint: Color) -> some View {
        HStack(spacing: 14) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(tint)
                    .frame(width: 36, height: 36)
                Image(systemName: systemImage)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body)
                    .foregroundStyle(.primary)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "chevron.right")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
    }
}

#Preview {
    ContentView()
}
