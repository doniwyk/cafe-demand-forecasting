import { createContext, useContext, useState, useCallback, useEffect, useMemo, type ReactNode } from "react";
import { authApi } from "@/features/auth/lib/api";

export interface User {
  id: number;
  name: string;
  email: string;
  avatar: string | null;
}

interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isInitialized: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const token = authApi.getToken();
    if (token) {
      authApi
        .me()
        .then((u) => setUser(u))
        .catch(() => {
          authApi.setToken(null);
          setUser(null);
        })
        .finally(() => setIsInitialized(true));
    } else {
      setIsInitialized(true);
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await authApi.login(email, password);
    authApi.setToken(res.access_token);
    setUser(res.user);
  }, []);

  const logout = useCallback(() => {
    authApi.setToken(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, isAuthenticated: user !== null, isInitialized, login, logout }),
    [user, isInitialized, login, logout]
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
}
