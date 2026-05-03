import { createRootRoute, Link, Outlet, useRouterState, useNavigate } from '@tanstack/react-router'
import { AppSidebar } from '@/components/app-sidebar'
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import { Separator } from '@/components/ui/separator'
import { TooltipProvider } from '@/components/ui/tooltip'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ModelProvider, useModelType, MODEL_TYPES, MODEL_LABELS } from '@/contexts/model-context'
import type { ModelType } from '@/contexts/model-context'
import { TourProvider } from '@/contexts/tour-context'
import { AppTour } from '@/components/app-tour'
import { BrainCircuitIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '@/contexts/auth-context'
import { useEffect } from 'react'

const routeTitleKeys: Record<string, string> = {
  '/': 'sidebar.dashboard',
  '/analytics': 'sidebar.analytics',
  '/forecasts': 'sidebar.salesForecast',
  '/materials/daily-need': 'materials.dailyMaterialRequirements',
  '/settings': 'sidebar.settings',
}

export const Route = createRootRoute({
  component: RootLayout,
})

function ModelSelector() {
  const { modelType, setModelType } = useModelType()

  return (
    <div className="flex items-center gap-2" data-tour="model-selector">
      <BrainCircuitIcon className="size-4 text-muted-foreground" />
      <Select value={modelType} onValueChange={(v) => setModelType(v as ModelType)}>
        <SelectTrigger size="sm" className="w-[160px]">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {MODEL_TYPES.map((type) => (
              <SelectItem key={type} value={type}>
                {MODEL_LABELS[type]}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  )
}

function RootLayout() {
  const routerState = useRouterState()
  const navigate = useNavigate()
  const pathname = routerState.location.pathname
  const { isAuthenticated, isInitialized } = useAuth()
  const { t } = useTranslation()
  const titleKey = routeTitleKeys[pathname] ?? 'sidebar.dashboard'
  const title = t(titleKey)

  const isLoginPage = pathname === '/login'

  useEffect(() => {
    if (!isInitialized) return
    if (!isAuthenticated && !isLoginPage) {
      navigate({ to: '/login', replace: true })
    }
    if (isAuthenticated && isLoginPage) {
      navigate({ to: '/', replace: true })
    }
  }, [isAuthenticated, isInitialized, isLoginPage, navigate])

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
              <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
                <SidebarTrigger className="-ml-1" />
                <Separator orientation="vertical" className="mr-2 h-4! self-center" />
                <Breadcrumb>
                  <BreadcrumbList>
                    <BreadcrumbItem>
                      <BreadcrumbLink render={<Link to="/" />}>{t('root.home')}</BreadcrumbLink>
                    </BreadcrumbItem>
                    {pathname !== '/' && (
                      <>
                        <BreadcrumbSeparator />
                        <BreadcrumbItem>
                          <BreadcrumbPage>{title}</BreadcrumbPage>
                        </BreadcrumbItem>
                      </>
                    )}
                  </BreadcrumbList>
                </Breadcrumb>
                <div className="ml-auto flex items-center gap-1">
                  <ModelSelector />
                  <AppTour />
                </div>
              </header>
              <Outlet />
            </SidebarInset>
          </SidebarProvider>
        </TooltipProvider>
      </TourProvider>
    </ModelProvider>
  )
}
