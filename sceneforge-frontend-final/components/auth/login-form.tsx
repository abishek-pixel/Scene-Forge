"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useUser } from "@/lib/user-context"

interface LoginFormProps {
  isSignUp: boolean
  onToggle: () => void
  onSuccess: () => void
  onForgotPassword: () => void
}

export default function LoginForm({ isSignUp, onToggle, onSuccess, onForgotPassword }: LoginFormProps) {
  const { login } = useUser()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    try {
      await login(email, password)
      setLoading(false)
      onSuccess()
    } catch (err) {
      setError("Invalid email or password")
      setLoading(false)
    }
  }

  return (
    <div className="glass-panel p-8 space-y-6">
      {/* Logo */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/20 border border-primary/30">
          <span className="text-xl font-bold text-primary">◆</span>
        </div>
        <h1 className="text-2xl font-bold text-foreground">SceneForge</h1>
        <p className="text-sm text-muted-foreground">{isSignUp ? "Create your account" : "Welcome back"}</p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm">
            {error}
          </div>
        )}

        <div className="space-y-2">
          <Label htmlFor="email" className="text-foreground">
            Email
          </Label>
          <Input
            id="email"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground"
            required
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="password" className="text-foreground">
              Password
            </Label>
            <button
              type="button"
              onClick={onForgotPassword}
              className="text-xs text-primary hover:text-primary/80 font-medium transition-colors"
            >
              Forgot?
            </button>
          </div>
          <Input
            id="password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="bg-input border-border/50 text-foreground placeholder:text-muted-foreground"
            required
          />
        </div>

        <Button type="submit" disabled={loading} className="w-full btn-primary">
          {loading ? "Loading..." : isSignUp ? "Sign Up" : "Sign In"}
        </Button>
      </form>

      {/* Divider */}
      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-border/30"></div>
        </div>
        <div className="relative flex justify-center text-xs uppercase">
          <span className="px-2 bg-card text-muted-foreground">Or continue with</span>
        </div>
      </div>

      {/* Social Login */}
      <div className="grid grid-cols-2 gap-3">
        <Button variant="outline" className="bg-card/50 border-border/50 hover:bg-card text-foreground">
          Google
        </Button>
        <Button variant="outline" className="bg-card/50 border-border/50 hover:bg-card text-foreground">
          GitHub
        </Button>
      </div>

      {/* Toggle */}
      <p className="text-center text-sm text-muted-foreground">
        {isSignUp ? "Already have an account?" : "Don't have an account?"}{" "}
        <button onClick={onToggle} className="text-primary hover:text-primary/80 font-medium transition-colors">
          {isSignUp ? "Sign In" : "Sign Up"}
        </button>
      </p>
    </div>
  )
}
