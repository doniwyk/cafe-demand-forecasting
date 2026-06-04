import { createRootRoute, Outlet, useRouterState, redirect } from '@tanstack/react-router'
import { AppSidebar } from '@/components/app-sidebar'
import { AppHeader } from '@/components/app-header'
import {
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ModelProvider } from '@/contexts/model-context'
import { TourProvider } from '@/contexts/tour-context'
import { useAuth } from '@/features/auth/contexts/auth-context'
import { getToken } from '@/lib/request'

export const Route = createRootRoute({
  beforeLoad: ({ location }) => {
    const token = getToken()
    const isLoginPage = location.pathname === '/login'

    if (!token && !isLoginPage) {
      throw redirect({ to: '/login' })
    }
    if (token && isLoginPage) {
      throw redirect({ to: '/' })
    }
  },
  component: RootLayout,
})

function RootLayout() {
  const routerState = useRouterState()
  const pathname = routerState.location.pathname
  const { isAuthenticated, isInitialized } = useAuth()

  const isLoginPage = pathname === '/login'

  if (!isInitialized) {
    return null
  }

  if (isLoginPage) {
    return <Outlet />
  }

  if (!isAuthenticated) {
    return null
  }

  return (
    <ModelProvider>
      <TourProvider>
        <TooltipProvider>
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
              <AppHeader />
              <Outlet />
            </SidebarInset>
          </SidebarProvider>
        </TooltipProvider>
      </TourProvider>
    </ModelProvider>
  )
}
