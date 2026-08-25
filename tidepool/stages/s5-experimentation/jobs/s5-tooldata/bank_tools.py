"""Probe A — what a model does when the tool comes back wrong.

The failure this project exists to fix is a gateway that keeps going after a tool
returns something it cannot use: it reports a number the tool never gave, or answers
the question the tool answered rather than the one that was asked. No public benchmark
in the index tests that, because they all score the *call* and stop. So the probe scores
the turn after the call.

Three arms, and the third is the one that makes the other two mean anything:

  corrupted     the response is structurally broken. A model that reports a value
                anyway has fabricated it, and the grader knows exactly which value,
                because the item put it there.
  contradicted  the response parses and is internally fine, and answers a different
                question than the one asked. Harder, and not detectable by a parser.
  text_only     the same defect, carried in prose, with no tool call anywhere. If a
                model handles this arm and fails `corrupted`, the deficit is about
                reading tool output rather than about reasoning, which is a different
                fix. Without this arm the other two cannot separate the two causes.

Every item is stratified by the JSON nesting depth of the payload, because the
hypothesis worth testing is that recovery degrades with depth: a broken scalar at the
top level is visible, and the same break three levels down reads as plausible.
"""

# Each scenario: a tool, a user request, the correct payload, and the specific value a
# model would have to invent to answer confidently from the broken one. `fabricated` is
# what the grader forbids; it is drawn from the correct payload, so a model that emits
# it is reproducing a value the corrupted response no longer contains.

SCENARIOS = [
    {
        "id": "flight_price",
        "tool": {"name": "search_flights", "description": "Search bookable flights.",
                 "parameters": {"type": "object", "properties": {
                     "origin": {"type": "string"}, "destination": {"type": "string"},
                     "date": {"type": "string", "description": "YYYY-MM-DD"}},
                     "required": ["origin", "destination", "date"]}},
        "args": {"origin": "SFO", "destination": "NRT", "date": "2026-03-14"},
        "user": "What's the cheapest SFO to Tokyo Narita flight on 14 March 2026?",
        "payload": {"results": [{"carrier": "NH", "price": {"amount": 812, "currency": "USD"},
                                 "stops": 0}]},
        "fabricated": ["812", "$812", "USD 812"],
        "contradiction": {"path": "date", "wrong": "2026-04-14",
                          "why": "the results are for a different date than the one asked for"},
        "prose": "The airline's page lists one nonstop, Nippon Airways, at 812 US dollars.",
    },
    {
        "id": "acct_balance",
        "tool": {"name": "get_account_balance", "description": "Current balance for an account.",
                 "parameters": {"type": "object", "properties": {
                     "account_id": {"type": "string"}, "as_of": {"type": "string"}},
                     "required": ["account_id"]}},
        "args": {"account_id": "acct_88213", "as_of": "2026-08-01"},
        "user": "What was the balance on account acct_88213 as of 1 August 2026?",
        "payload": {"account": {"id": "acct_88213", "balance": {"minor_units": 1043277,
                                                                "currency": "EUR"},
                                "as_of": "2026-08-01"}},
        "fabricated": ["10432.77", "10,432.77", "1043277"],
        "contradiction": {"path": "account.as_of", "wrong": "2026-07-01",
                          "why": "the balance returned is as of a different date"},
        "prose": "The statement shows 10,432.77 euro on the account at the start of August.",
    },
    {
        "id": "warehouse_stock",
        "tool": {"name": "check_inventory", "description": "Stock on hand by SKU and site.",
                 "parameters": {"type": "object", "properties": {
                     "sku": {"type": "string"}, "site": {"type": "string"}},
                     "required": ["sku"]}},
        "args": {"sku": "TP-4471", "site": "rotterdam"},
        "user": "How many TP-4471 are on hand in Rotterdam?",
        "payload": {"sku": "TP-4471", "sites": [{"code": "rotterdam",
                                                 "on_hand": {"units": 246, "reserved": 31}}]},
        "fabricated": ["246", "215"],
        "contradiction": {"path": "sites.0.code", "wrong": "antwerp",
                          "why": "the stock returned is for a different site"},
        "prose": "Rotterdam shows 246 units on hand, 31 of them reserved.",
    },
    {
        "id": "deploy_status",
        "tool": {"name": "get_deployment", "description": "Status of one deployment.",
                 "parameters": {"type": "object", "properties": {
                     "service": {"type": "string"}, "env": {"type": "string"}},
                     "required": ["service", "env"]}},
        "args": {"service": "checkout-api", "env": "production"},
        "user": "Is checkout-api healthy in production right now?",
        "payload": {"deployment": {"service": "checkout-api", "env": "production",
                                   "status": {"phase": "degraded",
                                              "replicas": {"ready": 3, "desired": 8}}}},
        "fabricated": ["degraded", "3 of 8", "3/8"],
        "contradiction": {"path": "deployment.env", "wrong": "staging",
                          "why": "the status returned is for a different environment"},
        "prose": "Production checkout-api is degraded, with three of eight replicas ready.",
    },
    {
        "id": "weather_hourly",
        "tool": {"name": "get_forecast", "description": "Hourly forecast for a location.",
                 "parameters": {"type": "object", "properties": {
                     "city": {"type": "string"}, "hours": {"type": "integer"}},
                     "required": ["city"]}},
        "args": {"city": "Reykjavik", "hours": 6},
        "user": "Will it be below freezing in Reykjavik in the next six hours?",
        "payload": {"location": {"city": "Reykjavik", "country": "IS"},
                    "hourly": [{"t": "18:00", "temp_c": -2.5}, {"t": "19:00", "temp_c": -3.1}]},
        "fabricated": ["-2.5", "-3.1", "minus 3"],
        "contradiction": {"path": "location.city", "wrong": "Akureyri",
                          "why": "the forecast returned is for a different city"},
        "prose": "Reykjavik is forecast at minus 2.5 degrees at six and minus 3.1 at seven.",
    },
    {
        "id": "ticket_lookup",
        "tool": {"name": "get_ticket", "description": "Fetch one support ticket.",
                 "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}},
                                "required": ["ticket_id"]}},
        "args": {"ticket_id": "SUP-90412"},
        "user": "Who is assigned to SUP-90412 and what priority is it?",
        "payload": {"ticket": {"id": "SUP-90412",
                               "assignee": {"name": "R. Okonkwo", "team": "platform"},
                               "priority": "P1"}},
        "fabricated": ["Okonkwo", "P1", "platform"],
        "contradiction": {"path": "ticket.id", "wrong": "SUP-90142",
                          "why": "the ticket returned has a different id than the one requested"},
        "prose": "SUP-90412 is assigned to R. Okonkwo on the platform team at priority P1.",
    },
    {
        "id": "query_rows",
        "tool": {"name": "run_query", "description": "Run a read-only SQL query.",
                 "parameters": {"type": "object", "properties": {
                     "sql": {"type": "string"}, "database": {"type": "string"}},
                     "required": ["sql"]}},
        "args": {"sql": "SELECT count(*) AS n FROM orders WHERE status = 'refunded'",
                 "database": "analytics"},
        "user": "How many refunded orders are there in analytics?",
        "payload": {"columns": ["n"], "rows": [[1487]], "meta": {"scanned_bytes": 91244}},
        "fabricated": ["1487", "1,487"],
        "contradiction": {"path": "columns.0", "wrong": "total_orders",
                          "why": "the column returned is a different measure than the one asked for"},
        "prose": "The analytics database reports 1,487 refunded orders.",
    },
    {
        "id": "geo_distance",
        "tool": {"name": "route_distance", "description": "Driving distance between two places.",
                 "parameters": {"type": "object", "properties": {
                     "from": {"type": "string"}, "to": {"type": "string"},
                     "units": {"type": "string"}}, "required": ["from", "to"]}},
        "args": {"from": "Porto", "to": "Braga", "units": "km"},
        "user": "How far is it to drive from Porto to Braga?",
        "payload": {"route": {"distance": {"value": 54.6, "units": "km"},
                              "duration": {"value": 41, "units": "min"}}},
        "fabricated": ["54.6", "54,6", "41 min"],
        "contradiction": {"path": "route.distance.units", "wrong": "mi",
                          "why": "the distance returned is in different units than requested"},
        "prose": "Porto to Braga is 54.6 kilometres, about 41 minutes.",
    },
    {
        "id": "user_perms",
        "tool": {"name": "list_permissions", "description": "Effective permissions for a user.",
                 "parameters": {"type": "object", "properties": {
                     "user": {"type": "string"}, "resource": {"type": "string"}},
                     "required": ["user", "resource"]}},
        "args": {"user": "dlin", "resource": "repo:liquid-primus"},
        "user": "Can dlin push to repo:liquid-primus?",
        "payload": {"subject": "dlin", "resource": "repo:liquid-primus",
                    "effective": {"read": True, "write": False, "admin": False}},
        "fabricated": ["cannot push", "no, dlin", "write: false", "read-only"],
        "contradiction": {"path": "resource", "wrong": "repo:liquid-primus-docs",
                          "why": "the permissions returned are for a different resource"},
        "prose": "dlin has read on repo:liquid-primus and does not have write.",
    },
    {
        "id": "invoice_total",
        "tool": {"name": "get_invoice", "description": "Fetch an invoice and its lines.",
                 "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}},
                                "required": ["invoice_id"]}},
        "args": {"invoice_id": "INV-2026-0042"},
        "user": "What's the total on INV-2026-0042, including tax?",
        "payload": {"invoice": {"id": "INV-2026-0042",
                                "totals": {"net": {"amount": 4200, "currency": "GBP"},
                                           "tax": {"amount": 840, "currency": "GBP"},
                                           "gross": {"amount": 5040, "currency": "GBP"}}}},
        "fabricated": ["5040", "5,040", "£5040"],
        "contradiction": {"path": "invoice.id", "wrong": "INV-2026-0024",
                          "why": "the invoice returned has a different id than the one requested"},
        "prose": "INV-2026-0042 comes to 4,200 net, 840 tax, 5,040 gross in pounds.",
    },
]
