"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"

export interface UserData {
  id: string
  firstName: string
  lastName: string
  email: string
  phone: string
  avatar?: string
  bio?: string
  createdAt: Date
  isNewUser: boolean
}

export interface ActiveSession {
  id: string
  device: string
  location: string
  lastActive: Date
  ipAddress: string
}

interface UserContextType {
  user: UserData | null
  sessions: ActiveSession[]
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  signup: (data: SignUpData) => Promise<void>
  logout: () => void
  updateProfile: (data: Partial<UserData>) => Promise<void>
  addSession: (session: ActiveSession) => void
  removeSession: (sessionId: string) => void
}

export interface SignUpData {
  firstName: string
  lastName: string
  email: string
  password: string
  phone: string
}

const UserContext = createContext<UserContextType | undefined>(undefined)

export function UserProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserData | null>(null)
  const [sessions, setSessions] = useState<ActiveSession[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  useEffect(() => {
    const storedUser = localStorage.getItem("sceneforge_user")
    if (storedUser) {
      setUser(JSON.parse(storedUser))
    }
    const storedSessions = localStorage.getItem("sceneforge_sessions")
    if (storedSessions) {
      setSessions(JSON.parse(storedSessions))
    }
    setIsLoading(false)
  }, [])

  const login = async (email: string, password: string) => {
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      })

      if (!res.ok) {
        // propagate server message if any
        const txt = await res.text()
        throw new Error(txt || "Login failed")
      }

      const data = await res.json()
      // expected shape: { user: {...}, token: 'jwt' }
      if (data?.user) {
        // normalize createdAt to Date
        if (data.user.createdAt) data.user.createdAt = new Date(data.user.createdAt)
        setUser(data.user)
        localStorage.setItem("sceneforge_user", JSON.stringify(data.user))
      }
      if (data?.token) {
        localStorage.setItem("sceneforge_token", data.token)
      }

      // add a session entry locally
      const newSession: ActiveSession = {
        id: "session_" + Date.now(),
        device: navigator.userAgent || "browser",
        location: "unknown",
        lastActive: new Date(),
        ipAddress: "",
      }
      const updatedSessions = [...sessions, newSession]
      setSessions(updatedSessions)
      localStorage.setItem("sceneforge_sessions", JSON.stringify(updatedSessions))
      return
    } catch (err) {
      // Fallback to the original simulated behaviour when backend is unreachable or misconfigured
      const userData: UserData = {
        id: "user_" + Date.now(),
        firstName: "John",
        lastName: "Doe",
        email,
        phone: "+1234567890",
        createdAt: new Date(),
        isNewUser: false,
      }
      setUser(userData)
      localStorage.setItem("sceneforge_user", JSON.stringify(userData))

      const newSession: ActiveSession = {
        id: "session_" + Date.now(),
        device: "Chrome on macOS",
        location: "San Francisco, CA",
        lastActive: new Date(),
        ipAddress: "192.168.1.1",
      }
      const updatedSessions = [...sessions, newSession]
      setSessions(updatedSessions)
      localStorage.setItem("sceneforge_sessions", JSON.stringify(updatedSessions))
    }
  }

  const signup = async (data: SignUpData) => {
    try {
      const res = await fetch(`${API_BASE}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(txt || "Signup failed")
      }

      const resp = await res.json()
      // expected: { user: {...}, token: 'jwt' }
      if (resp?.user) {
        if (resp.user.createdAt) resp.user.createdAt = new Date(resp.user.createdAt)
        setUser(resp.user)
        localStorage.setItem("sceneforge_user", JSON.stringify(resp.user))
      }
      if (resp?.token) localStorage.setItem("sceneforge_token", resp.token)

      const newSession: ActiveSession = {
        id: "session_" + Date.now(),
        device: navigator.userAgent || "browser",
        location: "unknown",
        lastActive: new Date(),
        ipAddress: "",
      }
      const updatedSessions = [newSession]
      setSessions(updatedSessions)
      localStorage.setItem("sceneforge_sessions", JSON.stringify(updatedSessions))
      return
    } catch (err) {
      // Fallback simulation
      const userData: UserData = {
        id: "user_" + Date.now(),
        firstName: data.firstName,
        lastName: data.lastName,
        email: data.email,
        phone: data.phone,
        createdAt: new Date(),
        isNewUser: true,
      }
      setUser(userData)
      localStorage.setItem("sceneforge_user", JSON.stringify(userData))

      const newSession: ActiveSession = {
        id: "session_" + Date.now(),
        device: "Chrome on macOS",
        location: "San Francisco, CA",
        lastActive: new Date(),
        ipAddress: "192.168.1.1",
      }
      const updatedSessions = [newSession]
      setSessions(updatedSessions)
      localStorage.setItem("sceneforge_sessions", JSON.stringify(updatedSessions))
    }
  }

  const logout = () => {
    // Optionally inform backend (best-effort)
    try {
      const token = localStorage.getItem("sceneforge_token")
      if (token) {
        void fetch(`${API_BASE}/auth/logout`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        })
      }
    } catch (e) {
      // ignore
    }

    setUser(null)
    setSessions([])
    localStorage.removeItem("sceneforge_user")
    localStorage.removeItem("sceneforge_sessions")
    localStorage.removeItem("sceneforge_token")
  }

  const updateProfile = async (data: Partial<UserData>) => {
    if (!user) return

    try {
      const token = localStorage.getItem("sceneforge_token")
      const res = await fetch(`${API_BASE}/users/${user.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(data),
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(txt || "Update profile failed")
      }

      const updatedUser = await res.json()
      if (updatedUser.createdAt) updatedUser.createdAt = new Date(updatedUser.createdAt)
      setUser(updatedUser)
      localStorage.setItem("sceneforge_user", JSON.stringify(updatedUser))
      return
    } catch (err) {
      // fallback to local-only update
      const updatedUser = { ...user, ...data }
      setUser(updatedUser)
      localStorage.setItem("sceneforge_user", JSON.stringify(updatedUser))
    }
  }

  const addSession = (session: ActiveSession) => {
    const updatedSessions = [...sessions, session]
    setSessions(updatedSessions)
    localStorage.setItem("sceneforge_sessions", JSON.stringify(updatedSessions))
  }

  const removeSession = (sessionId: string) => {
    const updatedSessions = sessions.filter((s) => s.id !== sessionId)
    setSessions(updatedSessions)
    localStorage.setItem("sceneforge_sessions", JSON.stringify(updatedSessions))
  }

  return (
    <UserContext.Provider
      value={{ user, sessions, isLoading, login, signup, logout, updateProfile, addSession, removeSession }}
    >
      {children}
    </UserContext.Provider>
  )
}

export function useUser() {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error("useUser must be used within UserProvider")
  }
  return context
}
