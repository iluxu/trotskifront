export async function onRequestGet({ request, env }) {
  const stripeKey = env.STRIPE_SECRET_KEY;
  const priceId = env.STRIPE_PRICE_ID || 'price_1ShpGNEEnbmnBN8hrflIMoh0';
  const signingSecret = env.LICENSE_SIGNING_SECRET;

  if (!stripeKey) return json({ error: 'Missing STRIPE_SECRET_KEY' }, 500);
  if (!signingSecret) return json({ error: 'Missing LICENSE_SIGNING_SECRET' }, 500);

  const url = new URL(request.url);
  const sessionId = url.searchParams.get('session_id');
  if (!sessionId) return json({ error: 'Missing session_id' }, 400);

  const stripeUrl = new URL(`https://api.stripe.com/v1/checkout/sessions/${sessionId}`);
  stripeUrl.searchParams.set('expand[]', 'line_items');

  const res = await fetch(stripeUrl.toString(), {
    headers: { Authorization: `Bearer ${stripeKey}` }
  });

  const data = await res.json();
  if (!res.ok) return json({ error: data?.error?.message || 'Stripe error' }, res.status);

  const paymentOk = data.payment_status === 'paid' && data.status === 'complete';
  const firstItem = data.line_items?.data?.[0];
  const priceMatch = !priceId || firstItem?.price?.id === priceId;

  if (!paymentOk || !priceMatch) {
    return json({ error: 'Checkout not complete' }, 402);
  }

  const license = await buildLicense(sessionId, signingSecret);

  return json({
    license,
    customer_email: data.customer_details?.email || null,
    subscription_id: data.subscription || null
  });
}

async function buildLicense(sessionId, secret) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(sessionId));
  const hex = bufferToHex(sig).toUpperCase();
  const raw = hex.slice(0, 20);
  return raw.match(/.{1,4}/g).join('-');
}

function bufferToHex(buffer) {
  const bytes = new Uint8Array(buffer);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store'
    }
  });
}
