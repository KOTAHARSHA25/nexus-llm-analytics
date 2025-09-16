"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { History, HelpCircle, Settings, FileText, Clock } from "lucide-react"

const recentQueries = [
  {
    query: "Show revenue trends",
    timestamp: "2 minutes ago",
    status: "completed",
  },
  {
    query: "Analyze customer data",
    timestamp: "1 hour ago",
    status: "completed",
  },
  {
    query: "Export monthly report",
    timestamp: "3 hours ago",
    status: "completed",
  },
]

export function Sidebar() {
  return (
    <div className="h-full bg-sidebar border-r border-sidebar-border p-4 space-y-6">
      {/* Recent Queries */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <History className="h-4 w-4" />
            Recent Queries
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {recentQueries.map((item, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs font-medium leading-relaxed">{item.query}</p>
                <Badge variant="outline" className="text-xs">
                  {item.status}
                </Badge>
              </div>
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                {item.timestamp}
              </div>
              {index < recentQueries.length - 1 && <Separator />}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
            <FileText className="h-4 w-4" />
            View All Reports
          </Button>
          <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
            <Settings className="h-4 w-4" />
            Settings
          </Button>
          <Button variant="ghost" size="sm" className="w-full justify-start gap-2">
            <HelpCircle className="h-4 w-4" />
            Help & Support
          </Button>
        </CardContent>
      </Card>

      {/* Tips */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">💡 Pro Tip</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Try asking specific questions like "What are the top 5 customers by revenue?" for more targeted insights.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
