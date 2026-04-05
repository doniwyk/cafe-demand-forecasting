import { createRootRoute, Link, Outlet, useRouterState } from '@tanstack/react-router'
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
import { BrainCircuitIcon } from 'lucide-react'

const routeTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/analytics': 'Analytics',
  '/forecasts': 'Sales Forecast',
  '/settings': 'Settings',
}

export const Route = createRootRoute({
  component: RootLayout,
})

function ModelSelector() {
  const { modelType, setModelType } = useModelType()

  return (
    <div className="flex items-center gap-2">
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
  const pathname = routerState.location.pathname
  const title = routeTitles[pathname] ?? 'Dashboard'

  return (
    <ModelProvider>
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
                    <BreadcrumbLink render={<Link to="/" />}>Home</BreadcrumbLink>
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
              <div className="ml-auto">
                <ModelSelector />
              </div>
            </header>
            <Outlet />
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </ModelProvider>
  )
}
