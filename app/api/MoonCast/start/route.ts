import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_POD_URL = process.env.RUNPOD_POD_URL;

export async function POST(request: NextRequest) {
  try {
    const { topic, duration = 1, seed } = await request.json();

    if (!topic) {
      return NextResponse.json({ error: 'Topic is required' }, { status: 400 });
    }

    if (!RUNPOD_POD_URL) {
      return NextResponse.json({ error: 'RunPod pod URL not configured' }, { status: 500 });
    }

    const response = await fetch(`${RUNPOD_POD_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, duration, seed })
    });

    if (!response.ok) {
      throw new Error(`Pod API error: ${response.status}`);
    }

    const result = await response.json();
    return NextResponse.json(result);

  } catch (error) {
    console.error('Start generation error:', error);
    return NextResponse.json({ error: 'Failed to start generation' }, { status: 500 });
  }
} 