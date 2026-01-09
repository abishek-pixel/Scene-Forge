"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function VerifyEmailPage() {
  const router = useRouter()
  const [code, setCode] = useState("")
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    // Simulate API call
    setTimeout(() => {
      setLoading(false)
      router.push("/dashboard")
    }, 1000)
  }

  return (
    <main className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="glass-panel p-8 space-y-6">
          {/* Header */}
          <div className="text-center space-y-2">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/20 border border-primary/30">
              <span className="text-xl font-bold text-primary">✓</span>
            </div>
            <h1 className="text-2xl font-bold text-foreground">Verify Email</h1>
            <p className="text-sm text-muted-foreground">
              We've sent a verification code to your email. Enter it below to continue.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="code" className="text-sm font-medium text-foreground">
                Verification Code
              </label>
              <Input
                id="code"
                type="text"
                placeholder="000000"
                value={code}
                onChange={(e) => setCode(e.target.value.toUpperCase())}
                maxLength={6}
                className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground text-center text-lg tracking-widest"
                required
              />
            </div>

            <Button type="submit" disabled={loading} className="w-full btn-primary">
              {loading ? "Verifying..." : "Verify Email"}
            </Button>
          </form>

          {/* Resend */}
          <p className="text-center text-sm text-muted-foreground">
            Didn't receive the code?{" "}
            <button className="text-primary hover:text-primary/80 font-medium transition-colors">Resend</button>
          </p>
        </div>
      </div>
    </main>
  )
}
