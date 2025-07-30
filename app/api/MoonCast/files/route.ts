import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_POD_URL = process.env.RUNPOD_POD_URL || "https://7b59ef18fecc.ngrok-free.app";

export async function GET(request: NextRequest) {
  try {
    if (!RUNPOD_POD_URL) {
      return NextResponse.json({ error: 'RunPod pod URL not configured' }, { status: 500 });
    }

    const response = await fetch(`${RUNPOD_POD_URL}/files`);

    if (!response.ok) {
      throw new Error(`Pod API error: ${response.status}`);
    }

    const result = await response.json();
    return NextResponse.json(result);

  } catch (error) {
    console.error('Files list error:', error);
    return NextResponse.json({ error: 'Failed to list files' }, { status: 500 });
  }
} 