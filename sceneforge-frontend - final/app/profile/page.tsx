"use client"

import AppLayout from "@/components/layout/app-layout"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { User, Lock, Bell, LogOut, Copy, Eye, EyeOff, Trash2, Plus, CreditCard, Shield } from "lucide-react"
import { useState } from "react"
import { useUser } from "@/lib/user-context"

export default function ProfilePage() {
  const { user, sessions, updateProfile, removeSession, logout } = useUser()
  const [activeTab, setActiveTab] = useState("profile")
  const [showPassword, setShowPassword] = useState(false)
  const [apiKeys, setApiKeys] = useState([
    {
      id: 1,
      name: "Production Key",
      key: "sk_live_••••••••••••••••",
      created: "2 months ago",
      lastUsed: "2 hours ago",
    },
    { id: 2, name: "Development Key", key: "sk_test_••••••••••••••••", created: "1 month ago", lastUsed: "1 day ago" },
  ])
  const [showNewKeyForm, setShowNewKeyForm] = useState(false)
  const [editMode, setEditMode] = useState(false)
  const [formData, setFormData] = useState({
    firstName: user?.firstName || "",
    lastName: user?.lastName || "",
    email: user?.email || "",
    phone: user?.phone || "",
    bio: user?.bio || "",
  })

  const tabs = [
    { id: "profile", label: "Profile", icon: User },
    { id: "security", label: "Security", icon: Lock },
    { id: "notifications", label: "Notifications", icon: Bell },
    { id: "api", label: "API Keys", icon: Shield },
    { id: "billing", label: "Billing", icon: CreditCard },
  ]

  const handleSaveProfile = async () => {
    await updateProfile({
      firstName: formData.firstName,
      lastName: formData.lastName,
      email: formData.email,
      phone: formData.phone,
      bio: formData.bio,
    })
    setEditMode(false)
  }

  return (
    <AppLayout>
      <div className="p-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Settings</h1>
          <p className="text-muted-foreground mt-2">Manage your account and preferences.</p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b border-border/30 overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-primary text-primary"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Profile Tab */}
        {activeTab === "profile" && (
          <div className="space-y-6 max-w-2xl">
            {/* Profile Info */}
            <Card className="glass-panel p-6">
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-20 h-20 rounded-lg bg-primary/20 flex items-center justify-center">
                    <User className="w-10 h-10 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-foreground">
                      {user?.firstName} {user?.lastName}
                    </h2>
                    <p className="text-sm text-muted-foreground">{user?.email}</p>
                    <Button
                      size="sm"
                      variant="outline"
                      className="mt-2 bg-card/50 border-border/50 hover:bg-card text-foreground"
                    >
                      Change Avatar
                    </Button>
                  </div>
                </div>

                <div className="border-t border-border/30 pt-6 space-y-4">
                  {editMode ? (
                    <>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium text-foreground">First Name</label>
                          <input
                            type="text"
                            value={formData.firstName}
                            onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                            className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                          />
                        </div>
                        <div>
                          <label className="text-sm font-medium text-foreground">Last Name</label>
                          <input
                            type="text"
                            value={formData.lastName}
                            onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                            className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="text-sm font-medium text-foreground">Email</label>
                        <input
                          type="email"
                          value={formData.email}
                          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                          className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                        />
                      </div>

                      <div>
                        <label className="text-sm font-medium text-foreground">Phone</label>
                        <input
                          type="tel"
                          value={formData.phone}
                          onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                          className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                        />
                      </div>

                      <div>
                        <label className="text-sm font-medium text-foreground">Bio</label>
                        <textarea
                          value={formData.bio}
                          onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
                          className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                          rows={3}
                        />
                      </div>

                      <div className="flex gap-3 pt-4">
                        <Button onClick={handleSaveProfile} className="btn-primary">
                          Save Changes
                        </Button>
                        <Button
                          onClick={() => setEditMode(false)}
                          variant="outline"
                          className="bg-card/50 border-border/50 hover:bg-card text-foreground"
                        >
                          Cancel
                        </Button>
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <label className="text-sm font-medium text-muted-foreground">Full Name</label>
                        <p className="text-foreground mt-1">
                          {user?.firstName} {user?.lastName}
                        </p>
                      </div>

                      <div>
                        <label className="text-sm font-medium text-muted-foreground">Email</label>
                        <p className="text-foreground mt-1">{user?.email}</p>
                      </div>

                      <div>
                        <label className="text-sm font-medium text-muted-foreground">Phone</label>
                        <p className="text-foreground mt-1">{user?.phone}</p>
                      </div>

                      <div>
                        <label className="text-sm font-medium text-muted-foreground">Bio</label>
                        <p className="text-foreground mt-1">{user?.bio || "No bio added yet"}</p>
                      </div>

                      <div className="flex gap-3 pt-4">
                        <Button onClick={() => setEditMode(true)} className="btn-primary">
                          Edit Profile
                        </Button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Security Tab */}
        {activeTab === "security" && (
          <div className="space-y-6 max-w-2xl">
            {/* Change Password */}
            <Card className="glass-panel p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Change Password</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-foreground">Current Password</label>
                  <div className="relative mt-2">
                    <input
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="w-full px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                    />
                    <button
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="text-sm font-medium text-foreground">New Password</label>
                  <input
                    type="password"
                    placeholder="••••••••"
                    className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-foreground">Confirm Password</label>
                  <input
                    type="password"
                    placeholder="••••••••"
                    className="w-full mt-2 px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                </div>

                <Button className="btn-primary">Update Password</Button>
              </div>
            </Card>

            {/* Two-Factor Authentication */}
            <Card className="glass-panel p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Two-Factor Authentication</h3>
                  <p className="text-sm text-muted-foreground mt-1">Add an extra layer of security to your account</p>
                </div>
                <Button variant="outline" className="bg-card/50 border-border/50 hover:bg-card text-foreground">
                  Enable 2FA
                </Button>
              </div>
            </Card>

            {/* Active Sessions */}
            <Card className="glass-panel p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Active Sessions</h3>
              {sessions.length === 0 ? (
                <p className="text-muted-foreground text-sm">No active sessions</p>
              ) : (
                <div className="space-y-3">
                  {sessions.map((session) => (
                    <div key={session.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <div>
                        <p className="font-medium text-foreground">{session.device}</p>
                        <p className="text-xs text-muted-foreground">
                          {session.location} • {session.lastActive.toLocaleString()}
                        </p>
                      </div>
                      <Button
                        size="sm"
                        onClick={() => removeSession(session.id)}
                        variant="outline"
                        className="bg-destructive/10 border-destructive/30 hover:bg-destructive/20 text-destructive"
                      >
                        Logout
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>
        )}

        {/* Notifications Tab */}
        {activeTab === "notifications" && (
          <div className="space-y-6 max-w-2xl">
            <Card className="glass-panel p-6">
              <div className="space-y-4">
                {[
                  { title: "Email Notifications", description: "Get updates about your scenes" },
                  { title: "Processing Complete", description: "Notify when scenes are ready" },
                  { title: "Weekly Summary", description: "Get a summary of your activity" },
                  { title: "New Features", description: "Be notified about new features" },
                  { title: "Security Alerts", description: "Important security notifications" },
                ].map((item, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/20 transition-colors"
                  >
                    <div>
                      <p className="font-medium text-foreground">{item.title}</p>
                      <p className="text-sm text-muted-foreground">{item.description}</p>
                    </div>
                    <input type="checkbox" defaultChecked className="w-5 h-5 rounded" />
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* API Keys Tab */}
        {activeTab === "api" && (
          <div className="space-y-6 max-w-2xl">
            {/* API Keys List */}
            <Card className="glass-panel p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-foreground">API Keys</h3>
                <Button size="sm" className="btn-primary" onClick={() => setShowNewKeyForm(!showNewKeyForm)}>
                  <Plus className="w-4 h-4 mr-2" />
                  New Key
                </Button>
              </div>

              {showNewKeyForm && (
                <div className="p-4 rounded-lg bg-muted/50 mb-6 space-y-3">
                  <input
                    type="text"
                    placeholder="Key name (e.g., Production)"
                    className="w-full px-4 py-2 bg-input border border-border/50 rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                  <div className="flex gap-2">
                    <Button className="flex-1 btn-primary">Create Key</Button>
                    <Button
                      variant="outline"
                      onClick={() => setShowNewKeyForm(false)}
                      className="flex-1 bg-card/50 border-border/50 hover:bg-card text-foreground"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              <div className="space-y-3">
                {apiKeys.map((key) => (
                  <div key={key.id} className="p-4 rounded-lg bg-muted/50 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="font-medium text-foreground">{key.name}</p>
                      <Button
                        size="sm"
                        variant="outline"
                        className="bg-destructive/10 border-destructive/30 hover:bg-destructive/20 text-destructive"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 text-xs text-muted-foreground bg-background/50 px-3 py-2 rounded">
                        {key.key}
                      </code>
                      <Button
                        size="sm"
                        variant="outline"
                        className="bg-card/50 border-border/50 hover:bg-card text-foreground"
                      >
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="flex gap-4 text-xs text-muted-foreground">
                      <span>Created {key.created}</span>
                      <span>Last used {key.lastUsed}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* API Documentation */}
            <Card className="glass-panel p-6 border-primary/30 bg-primary/5">
              <h3 className="font-semibold text-foreground mb-2">API Documentation</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Learn how to integrate SceneForge with your applications.
              </p>
              <Button variant="outline" className="bg-card/50 border-border/50 hover:bg-card text-foreground">
                View Documentation
              </Button>
            </Card>
          </div>
        )}

        {/* Billing Tab */}
        {activeTab === "billing" && (
          <div className="space-y-6 max-w-2xl">
            {/* Current Plan */}
            <Card className="glass-panel p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Current Plan</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 rounded-lg bg-primary/10 border border-primary/30">
                  <div>
                    <p className="font-semibold text-foreground">Pro Plan</p>
                    <p className="text-sm text-muted-foreground">$29/month</p>
                  </div>
                  <Button variant="outline" className="bg-card/50 border-border/50 hover:bg-card text-foreground">
                    Manage Plan
                  </Button>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-muted-foreground">Processing Time</p>
                    <p className="font-semibold text-foreground mt-1">Unlimited</p>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/50">
                    <p className="text-muted-foreground">Storage</p>
                    <p className="font-semibold text-foreground mt-1">1 TB</p>
                  </div>
                </div>
              </div>
            </Card>

            {/* Billing History */}
            <Card className="glass-panel p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Billing History</h3>
              <div className="space-y-2">
                {[
                  { date: "Oct 29, 2024", amount: "$29.00", status: "Paid" },
                  { date: "Sep 29, 2024", amount: "$29.00", status: "Paid" },
                  { date: "Aug 29, 2024", amount: "$29.00", status: "Paid" },
                ].map((invoice, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/20 transition-colors"
                  >
                    <div>
                      <p className="font-medium text-foreground">{invoice.date}</p>
                      <p className="text-sm text-muted-foreground">{invoice.amount}</p>
                    </div>
                    <span className="text-sm text-green-500">{invoice.status}</span>
                  </div>
                ))}
              </div>
            </Card>

            {/* Payment Method */}
            <Card className="glass-panel p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Payment Method</h3>
              <div className="p-4 rounded-lg bg-muted/50 flex items-center justify-between">
                <div>
                  <p className="font-medium text-foreground">Visa ending in 4242</p>
                  <p className="text-sm text-muted-foreground">Expires 12/26</p>
                </div>
                <Button variant="outline" className="bg-card/50 border-border/50 hover:bg-card text-foreground">
                  Update
                </Button>
              </div>
            </Card>
          </div>
        )}

        {/* Danger Zone */}
        <Card className="glass-panel p-6 border-destructive/30 max-w-2xl">
          <h3 className="text-lg font-semibold text-destructive mb-4">Danger Zone</h3>
          <div className="space-y-3">
            <Button
              onClick={() => {
                // Logout from all devices
                logout()
              }}
              variant="outline"
              className="w-full bg-destructive/10 border-destructive/30 hover:bg-destructive/20 text-destructive justify-start"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout from All Devices
            </Button>
            <Button
              variant="outline"
              className="w-full bg-destructive/10 border-destructive/30 hover:bg-destructive/20 text-destructive justify-start"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Delete Account
            </Button>
          </div>
        </Card>
      </div>
    </AppLayout>
  )
}
