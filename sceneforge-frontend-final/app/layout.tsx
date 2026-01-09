import type { Metadata } from "next"
import { Geist } from "next/font/google"
import { UserProvider } from "@/lib/user-context"
import { AuthProvider } from "@/lib/auth-context"
import "./globals.css"

const geist = Geist({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "SceneForge - 3D Scene Generation",
  description: "Transform videos and images into interactive 3D scenes and assets",
  generator: "v0.app",
}

interface RootLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({
  children,
}: RootLayoutProps) {
  return (
    <html lang="en" className="dark">
      <body className={`${geist.className} antialiased`}>
        <AuthProvider>
          <UserProvider>{children}</UserProvider>
        </AuthProvider>
      </body>
    </html>
  )
}
