"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import LoginForm from "@/components/auth/login-form"
import SignUpForm from "@/components/auth/signup-form"
import ForgotPasswordForm from "@/components/auth/forgot-password-form"

type AuthMode = "login" | "signup" | "forgot"

export default function AuthPage() {
  const router = useRouter()
  const [mode, setMode] = useState<AuthMode>("login")

  const handleLoginSuccess = () => {
    router.push("/dashboard")
  }

  const handleSignUpSuccess = () => {
    router.push("/auth/verify-email")
  }

  const handleForgotPasswordSubmit = () => {
    setMode("login")
  }

  return (
    <main className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {mode === "login" && (
          <LoginForm
            onToggle={() => setMode("signup")}
            onForgotPassword={() => setMode("forgot")}
            onSuccess={handleLoginSuccess}
          />
        )}
        {mode === "signup" && <SignUpForm onToggle={() => setMode("login")} onSuccess={handleSignUpSuccess} />}
        {mode === "forgot" && (
          <ForgotPasswordForm onBack={() => setMode("login")} onSuccess={handleForgotPasswordSubmit} />
        )}
      </div>
    </main>
  )
}
