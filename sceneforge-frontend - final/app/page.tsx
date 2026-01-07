"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function Home() {
  const router = useRouter()
  const [isSignUp, setIsSignUp] = useState(false)

  useEffect(() => {
    router.push("/auth")
  }, [router])

  const handleLoginSuccess = () => {
    router.push("/dashboard")
  }

  return null
}
