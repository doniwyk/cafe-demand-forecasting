import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react'
import { api } from '@/lib/api'

export interface User {
  id: number
  name: string
  email: string
  avatar: string | null
}

interface AuthContextValue {
  user: User | null
  isAuthenticated: boolean
  isInitialized: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isInitialized, setIsInitialized] = useState(false)

  useEffect(() => {
    const token = api.auth.getToken()
    if (token) {
      api.auth.me()
        .then((u) => setUser(u))
        .catch(() => {
          api.auth.setToken(null)
          setUser(null)
        })
        .finally(() => setIsInitialized(true))
    } else {
      setIsInitialized(true)
    }
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const res = await api.auth.login(email, password)
    api.auth.setToken(res.access_token)
    setUser(res.user)
  }, [])

  const logout = useCallback(() => {
    api.auth.setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: user !== null, isInitialized, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used within an AuthProvider')
  return context
}
