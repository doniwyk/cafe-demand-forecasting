import { createFileRoute } from '@tanstack/react-router'
import { LoginPage } from '@/features/login/pages'

export const Route = createFileRoute('/login')({
  component: LoginPage,
})
