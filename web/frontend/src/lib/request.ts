const BASE = "/api";

export function getToken(): string | null {
  return localStorage.getItem("auth_token");
}

export function setToken(token: string | null) {
  if (token) {
    localStorage.setItem("auth_token", token);
  } else {
    localStorage.removeItem("auth_token");
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  options?: { timeoutMs?: number },
): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = options?.timeoutMs;
  const timeoutId =
    timeoutMs && timeoutMs > 0
      ? setTimeout(() => controller.abort("request-timeout"), timeoutMs)
      : undefined;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(`${BASE}${path}`, {
      headers,
      signal: controller.signal,
      ...init,
    });

    if (res.status === 401) {
      setToken(null);
      window.location.href = "/login";
      throw new Error("Session expired");
    }

    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.detail ?? `API error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

function buildQuery(params?: Record<string, string | number | undefined>): string {
  if (!params) return "";
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      searchParams.append(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

export const http = {
  get<T>(
    path: string,
    params?: Record<string, string | number | undefined>,
    options?: { timeoutMs?: number },
  ): Promise<T> {
    return request<T>(`${path}${buildQuery(params)}`, undefined, options);
  },
  post<T>(path: string, body?: unknown, options?: { timeoutMs?: number }): Promise<T> {
    return request<T>(
      path,
      { method: "POST", body: body ? JSON.stringify(body) : undefined },
      options,
    );
  },
};
