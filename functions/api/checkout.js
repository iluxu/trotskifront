export async function onRequest({ request, env }) {
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: corsHeaders()
    });
  }

  if (request.method !== 'POST') {
    return json({ error: 'Method not allowed' }, 405);
  }

  const stripeKey = env.STRIPE_SECRET_KEY;
  const priceId = (env.STRIPE_PRICE_ID || 'price_1ShpGNEEnbmnBN8hrflIMoh0').trim();

  if (!stripeKey) {
    return json({ error: 'Missing STRIPE_SECRET_KEY' }, 500);
  }

  const url = new URL(request.url);
  const origin = url.origin;

  const body = new URLSearchParams();
  body.set('mode', 'subscription');
  body.append('line_items[0][price]', priceId);
  body.append('line_items[0][quantity]', '1');
  body.set('success_url', `${origin}/success?session_id={CHECKOUT_SESSION_ID}`);
  body.set('cancel_url', `${origin}/?canceled=1`);
  body.append('payment_method_types[]', 'card');

  const res = await fetch('https://api.stripe.com/v1/checkout/sessions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${stripeKey}`,
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: body.toString()
  });

  const data = await res.json();
  if (!res.ok) {
    return json({ error: data?.error?.message || 'Stripe error' }, res.status);
  }

  return json({ url: data.url });
}

function json(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
      ...corsHeaders()
    }
  });
}

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  };
}
