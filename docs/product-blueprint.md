# AquaIQ Product Blueprint

## Vision

AquaIQ is an AI-assisted swimming coaching system built around four performance pillars: technique, training, race strategy, and mindset. The Phase 1 product is a private coach-first tool that proves measurable improvement with real swimmers before expanding into a broader coach product.

## Phase 1 MVP

The first build focuses on a production-shaped web application that a coach can use with swimmers immediately:

- Manage swimmer profiles and personal bests.
- Log training sessions with RPE, load, mood, sleep, notes, and optional video.
- Generate adaptive Base, Build, Peak, and Taper training plans.
- Create deterministic mock technique reports from video uploads.
- Analyze race splits and generate coaching insights.
- Capture mental check-ins and show trend-ready data.

## Architecture

- **Web:** Next.js App Router, TypeScript, Tailwind CSS, Recharts, Zustand.
- **API:** FastAPI, SQLAlchemy, Alembic, Pydantic, JWT auth.
- **Database:** PostgreSQL through Docker Compose locally, compatible with later Supabase deployment.
- **Storage:** Local uploads directory behind a storage-provider interface.
- **AI providers:** Mock/local providers by default, with provider contracts ready for future LLM and pose-estimation integrations.

## Phase 1 Success Criteria

- At least one coach can manage swimmers from a dashboard.
- A complete swimmer profile shows sessions, training plan, technique reports, race analyses, and mental check-ins.
- Video upload creates a realistic analysis lifecycle and report without paid services.
- The backend owns calculations and validation.
- Tests cover core calculations and the primary API flow.
