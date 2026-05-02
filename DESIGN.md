---
version: alpha
name: Archival Ink & Warm Copper
description: 深墨学术气质，暖铜强调色。用于新闻推荐系统的学术报告和演示。
colors:
  primary: "#1A1C1E"
  secondary: "#5F6B7A"
  accent: "#C7512E"
  neutral: "#F9F7F4"
  code-bg: "#F0EDE8"
typography:
  heading:
    fontFamily: Noto Serif CJK SC, IBM Plex Serif
    fontSize: 2rem
    fontWeight: 600
  body:
    fontFamily: Noto Serif CJK SC, IBM Plex Serif
    fontSize: 1rem
    fontWeight: 400
  code:
    fontFamily: IBM Plex Mono, monospace
    fontSize: 0.875rem
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
rounded:
  sm: 4px
  md: 8px
  lg: 16px
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 12px
  card:
    backgroundColor: "#FFFFFF"
    rounded: "{rounded.md}"
    padding: "{spacing.lg}"
