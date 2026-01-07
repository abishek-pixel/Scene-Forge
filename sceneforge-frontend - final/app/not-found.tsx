"use client"

import { useEffect } from "react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function NotFound() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-background/80 flex items-center justify-center p-4">
      <div className="max-w-md text-center">
        <div className="mb-8">
          <h1 className="text-6xl font-bold text-foreground mb-4">404</h1>
          <p className="text-2xl font-semibold text-foreground/80 mb-2">Page Not Found</p>
          <p className="text-foreground/60 mb-8">The page you're looking for doesn't exist or has been moved.</p>
        </div>
        
        <div className="space-y-3">
          <Link href="/">
            <Button className="w-full">Go to Home</Button>
          </Link>
          <Link href="/dashboard">
            <Button variant="outline" className="w-full">Go to Dashboard</Button>
          </Link>
        </div>

        <p className="text-sm text-foreground/40 mt-8">
          If this persists, please contact support
        </p>
      </div>
    </main>
  )
}
