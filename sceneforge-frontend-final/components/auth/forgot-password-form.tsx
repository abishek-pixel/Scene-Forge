"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface ForgotPasswordFormProps {
  onBack: () => void
  onSuccess: () => void
}

export default function ForgotPasswordForm({ onBack, onSuccess }: ForgotPasswordFormProps) {
  const [step, setStep] = useState<"contact" | "otp" | "reset" | "success">("contact")
  const [contactMethod, setContactMethod] = useState<"email" | "phone">("email")
  const [contact, setContact] = useState("")
  const [otp, setOtp] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleContactSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!contact.trim()) {
      setError(`Please enter your ${contactMethod}`)
      return
    }

    setLoading(true)
    // Simulate sending OTP
    setTimeout(() => {
      setLoading(false)
      setStep("otp")
    }, 1000)
  }

  const handleOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!otp.trim() || otp.length !== 6) {
      setError("Please enter a valid 6-digit OTP")
      return
    }

    setLoading(true)
    // Simulate OTP verification
    setTimeout(() => {
      setLoading(false)
      setStep("reset")
    }, 1000)
  }

  const handlePasswordReset = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")

    if (!newPassword.trim()) {
      setError("Please enter a new password")
      return
    }
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match")
      return
    }
    if (newPassword.length < 8) {
      setError("Password must be at least 8 characters")
      return
    }

    setLoading(true)
    // Simulate password reset
    setTimeout(() => {
      setLoading(false)
      setStep("success")
    }, 1000)
  }

  if (step === "success") {
    return (
      <div className="glass-panel p-8 space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-green-500/20 border border-green-500/30">
            <span className="text-xl font-bold text-green-500">✓</span>
          </div>
          <h1 className="text-2xl font-bold text-foreground">Password Reset</h1>
          <p className="text-sm text-muted-foreground">Your password has been successfully reset</p>
        </div>

        {/* Message */}
        <div className="p-4 rounded-lg bg-green-500/5 border border-green-500/20 text-sm text-foreground space-y-2">
          <p>You can now sign in with your new password.</p>
        </div>

        {/* Actions */}
        <Button onClick={onBack} className="w-full btn-primary">
          Back to Sign In
        </Button>
      </div>
    )
  }

  if (step === "contact") {
    return (
      <div className="glass-panel p-8 space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/20 border border-primary/30">
            <span className="text-xl font-bold text-primary">🔑</span>
          </div>
          <h1 className="text-2xl font-bold text-foreground">Reset Password</h1>
          <p className="text-sm text-muted-foreground">Choose how you'd like to receive your verification code</p>
        </div>

        {/* Contact Method Selection */}
        <div className="space-y-3">
          <button
            onClick={() => setContactMethod("email")}
            className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
              contactMethod === "email"
                ? "border-primary bg-primary/10"
                : "border-border/30 bg-card/50 hover:border-border/50"
            }`}
          >
            <p className="font-medium text-foreground">Email</p>
            <p className="text-sm text-muted-foreground">Receive code via email</p>
          </button>

          <button
            onClick={() => setContactMethod("phone")}
            className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
              contactMethod === "phone"
                ? "border-primary bg-primary/10"
                : "border-border/30 bg-card/50 hover:border-border/50"
            }`}
          >
            <p className="font-medium text-foreground">Phone</p>
            <p className="text-sm text-muted-foreground">Receive code via SMS</p>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleContactSubmit} className="space-y-4">
          {error && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
              {error}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="contact" className="text-foreground">
              {contactMethod === "email" ? "Email Address" : "Phone Number"}
            </Label>
            <Input
              id="contact"
              type={contactMethod === "email" ? "email" : "tel"}
              placeholder={contactMethod === "email" ? "you@example.com" : "+1 (555) 000-0000"}
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground"
              required
            />
          </div>

          <Button type="submit" disabled={loading} className="w-full btn-primary">
            {loading ? "Sending..." : "Send Verification Code"}
          </Button>
        </form>

        {/* Back Link */}
        <p className="text-center text-sm text-muted-foreground">
          <button onClick={onBack} className="text-primary hover:text-primary/80 font-medium transition-colors">
            Back to Sign In
          </button>
        </p>
      </div>
    )
  }

  if (step === "otp") {
    return (
      <div className="glass-panel p-8 space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/20 border border-primary/30">
            <span className="text-xl font-bold text-primary">📱</span>
          </div>
          <h1 className="text-2xl font-bold text-foreground">Verify Code</h1>
          <p className="text-sm text-muted-foreground">We sent a 6-digit code to your {contactMethod}</p>
        </div>

        {/* Form */}
        <form onSubmit={handleOtpSubmit} className="space-y-4">
          {error && (
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
              {error}
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="otp" className="text-foreground">
              Verification Code
            </Label>
            <Input
              id="otp"
              type="text"
              placeholder="000000"
              value={otp}
              onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))}
              maxLength={6}
              className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground text-center text-2xl tracking-widest"
              required
            />
          </div>

          <Button type="submit" disabled={loading} className="w-full btn-primary">
            {loading ? "Verifying..." : "Verify Code"}
          </Button>

          <Button
            type="button"
            variant="outline"
            onClick={() => setStep("contact")}
            className="w-full bg-card/50 border-border/50 hover:bg-card text-foreground"
          >
            Back
          </Button>
        </form>
      </div>
    )
  }

  return (
    <div className="glass-panel p-8 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/20 border border-primary/30">
          <span className="text-xl font-bold text-primary">🔐</span>
        </div>
        <h1 className="text-2xl font-bold text-foreground">Create New Password</h1>
        <p className="text-sm text-muted-foreground">Enter your new password below</p>
      </div>

      {/* Form */}
      <form onSubmit={handlePasswordReset} className="space-y-4">
        {error && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
            {error}
          </div>
        )}

        <div className="space-y-2">
          <Label htmlFor="newPassword" className="text-foreground">
            New Password
          </Label>
          <Input
            id="newPassword"
            type="password"
            placeholder="••••••••"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground"
            required
          />
          <p className="text-xs text-muted-foreground">At least 8 characters</p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="confirmPassword" className="text-foreground">
            Confirm Password
          </Label>
          <Input
            id="confirmPassword"
            type="password"
            placeholder="••••••••"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground"
            required
          />
        </div>

        <Button type="submit" disabled={loading} className="w-full btn-primary">
          {loading ? "Resetting..." : "Reset Password"}
        </Button>

        <Button
          type="button"
          variant="outline"
          onClick={() => setStep("contact")}
          className="w-full bg-card/50 border-border/50 hover:bg-card text-foreground"
        >
          Back
        </Button>
      </form>
    </div>
  )
}
