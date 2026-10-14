// ============================================================
// MuleGuard.AI — Neo4j Graph Initialisation
//
// Creates uniqueness constraints, indexes, and a test money ring
// (A → B → C → A) for validating the SCC ring-detection algorithm.
//
// How to run:
//   Option A — Neo4j Browser (http://localhost:7474)
//              Paste this file and run block by block.
//   Option B — cypher-shell:
//              docker exec -i mg_neo4j cypher-shell -u neo4j \
//                -p muleguard_neo4j < infrastructure/neo4j/seed_graph.cypher
// ============================================================

// ── CONSTRAINTS ──────────────────────────────────────────
CREATE CONSTRAINT account_token_unique IF NOT EXISTS
FOR (a:Account) REQUIRE a.token IS UNIQUE;

CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS
FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE;

// ── INDEXES ──────────────────────────────────────────────
// Supports temporal window queries (dormancy detection)
CREATE INDEX account_last_active_idx IF NOT EXISTS
FOR (a:Account) ON (a.last_active_ts);

// Supports Louvain community detection output queries
CREATE INDEX account_community_idx IF NOT EXISTS
FOR (a:Account) ON (a.community_id);

// Supports PageRank ranking queries
CREATE INDEX account_pagerank_idx IF NOT EXISTS
FOR (a:Account) ON (a.pagerank);

// ── TEST MONEY RING: A → B → C → A ──────────────────────
// A synthetic ring of 3 accounts used to verify SCC detection.
// All three nodes must appear in the same strongly-connected component.
// Remove these nodes once real training data is loaded.

MERGE (a:Account {token: 'TEST_ACCT_ALPHA_001'})
  SET a.label = 'test', a.is_test_node = true,
      a.created_at = datetime(), a.last_active_ts = datetime();

MERGE (b:Account {token: 'TEST_ACCT_BETA_002'})
  SET b.label = 'test', b.is_test_node = true,
      b.created_at = datetime(), b.last_active_ts = datetime();

MERGE (c:Account {token: 'TEST_ACCT_GAMMA_003'})
  SET c.label = 'test', c.is_test_node = true,
      c.created_at = datetime(), c.last_active_ts = datetime();

MATCH (a:Account {token: 'TEST_ACCT_ALPHA_001'}),
      (b:Account {token: 'TEST_ACCT_BETA_002'}),
      (c:Account {token: 'TEST_ACCT_GAMMA_003'})

// Edge A → B
MERGE (a)-[:SENT {
  tx_id: 'TEST_TXN_001', amount: 10000.00,
  timestamp: datetime() - duration('PT10M'), channel: 'upi'
}]->(b)

// Edge B → C
MERGE (b)-[:SENT {
  tx_id: 'TEST_TXN_002', amount: 9500.00,
  timestamp: datetime() - duration('PT7M'), channel: 'upi'
}]->(c)

// Edge C → A (closes the ring)
MERGE (c)-[:SENT {
  tx_id: 'TEST_TXN_003', amount: 9000.00,
  timestamp: datetime() - duration('PT5M'), channel: 'atm'
}]->(a);

// ── VERIFICATION ─────────────────────────────────────────
// Run these to confirm setup is correct:

SHOW CONSTRAINTS;
// Expected: 2+ rows

MATCH (n:Account) WHERE n.is_test_node = true RETURN n.token AS token;
// Expected: 3 rows

// Run SCC to verify all 3 test accounts share the same component:
// CALL gds.graph.project('test-ring-graph', 'Account', {SENT: {orientation: 'NATURAL'}});
// CALL gds.scc.stream('test-ring-graph') YIELD nodeId, componentId
// WITH gds.util.asNode(nodeId).token AS token, componentId
// WHERE token STARTS WITH 'TEST_ACCT'
// RETURN token, componentId ORDER BY componentId;
// CALL gds.graph.drop('test-ring-graph');

// ============================================================
// GDS ALGORITHM REFERENCE QUERIES
// These are the same queries run by scripts/seed_neo4j.py
// Run manually in Neo4j Browser after the full IBM AML load.
// ============================================================

// ── GDS: Project graph into in-memory catalog ────────────
CALL gds.graph.project(
  'muleguard_aml_graph',
  'Account',
  {SENT: {orientation: 'NATURAL'}}
)
YIELD graphName, nodeCount, relationshipCount, projectMillis;

// ── GDS: PageRank (write mode) ───────────────────────────
// Writes Account.pagerank (higher = more structurally important hub)

CALL gds.pageRank.write('muleguard_aml_graph', {
  maxIterations: 20,
  dampingFactor: 0.85,
  writeProperty: 'pagerank'
})
YIELD nodePropertiesWritten, ranIterations;

// ── GDS: Louvain Community Detection (write mode) ─────────
// Writes Account.community_id (accounts in same syndicate share a community)

CALL gds.louvain.write('muleguard_aml_graph', {
  writeProperty: 'community_id',
  maxLevels: 10,
  maxIterations: 10,
  tolerance: 0.0001
})
YIELD nodePropertiesWritten, communityCount, modularity;

// Verify: should return 100+ communities for the full IBM AML dataset
MATCH (a:Account) RETURN count(DISTINCT a.community_id) AS community_count;

// ── GDS: Strongly Connected Components (ring detection) ───
// Finds accounts in circular money flows (A→B→C→A patterns)
// Accounts in SCC components of size ≥ 3 are flagged as ring members.

// Step 1 — Stream SCC to find ring components:
CALL gds.scc.stream('muleguard_aml_graph')
YIELD nodeId, componentId
WITH componentId, count(nodeId) AS size
WHERE size >= 3
RETURN componentId, size ORDER BY size DESC LIMIT 20;

// Step 2 — Verify ring flag was written:
MATCH (a:Account) WHERE a.in_scc_ring = true RETURN count(a) AS ring_accounts;

// ── POST-GDS: Drop in-memory projection (free memory) ────
CALL gds.graph.drop('muleguard_aml_graph');

// ── QUERY: Top PageRank accounts (likely hubs/mules) ──────
// MATCH (a:Account)
// WHERE a.pagerank IS NOT NULL
// RETURN a.token, a.pagerank, a.community_id, a.in_scc_ring
// ORDER BY a.pagerank DESC LIMIT 20;

// ── QUERY: Top communities by flagged account volume ─────
// MATCH (a:Account)
// WHERE a.is_laundering_flag = 1
// WITH a.community_id AS community, count(a) AS flagged_in_community
// WHERE community IS NOT NULL
// RETURN community, flagged_in_community
// ORDER BY flagged_in_community DESC LIMIT 10;

// ── QUERY: Verify schema integrity ───────────────────────
// MATCH (a:Account) RETURN count(a) AS total_accounts;
// MATCH ()-[r:SENT]->() RETURN count(r) AS total_transactions;
// MATCH (a:Account) WHERE a.pagerank IS NULL RETURN count(a) AS missing_pagerank;
// MATCH (a:Account) WHERE a.community_id IS NULL RETURN count(a) AS missing_community;
// MATCH (a:Account) WHERE a.in_scc_ring IS NULL RETURN count(a) AS missing_scc;
