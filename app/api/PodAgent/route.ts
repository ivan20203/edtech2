import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { topic, guestNumber = 2, sessionId = "test" } = body;

    console.log('🚀 PodAgent API called with:', { topic, guestNumber, sessionId });

    if (!topic) {
      console.log('❌ Error: Topic is required');
      return NextResponse.json(
        { error: 'Topic is required' },
        { status: 400 }
      );
    }

    console.log('📞 Calling Flask service at http://127.0.0.1:8021/generate_podcast...');

    // Call the PodAgent Python service
    const response = await fetch('http://127.0.0.1:8021/generate_podcast', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topic,
        guest_number: guestNumber,
        session_id: sessionId,
      }),
    });

    console.log('📡 Flask service response status:', response.status);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.log('❌ Flask service error:', errorData);
      return NextResponse.json(
        { error: 'Failed to generate podcast', details: errorData },
        { status: response.status }
      );
    }

    const result = await response.json();
    console.log('✅ Podcast generation successful:', result);
    return NextResponse.json(result);

  } catch (error) {
    console.error('💥 PodAgent API error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  console.log('📋 PodAgent API GET request');
  return NextResponse.json(
    { message: 'PodAgent API is running. Use POST with topic, guestNumber, and sessionId parameters.' },
    { status: 200 }
  );
}
  