/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Empty turbopack config to use Turbopack (Next.js 16 default) without errors
  turbopack: {},
}

module.exports = nextConfig
