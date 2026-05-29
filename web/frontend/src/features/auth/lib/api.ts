import { http, getToken, setToken } from "@/lib/request";

export const authApi = {
  login(email: string, password: string) {
    return http.post<{
      access_token: string;
      token_type: string;
      user: { id: number; email: string; name: string; avatar: string | null };
    }>("/auth/login", { email, password });
  },
  me() {
    return http.get<{ id: number; email: string; name: string; avatar: string | null }>("/auth/me");
  },
  getToken,
  setToken,
};
