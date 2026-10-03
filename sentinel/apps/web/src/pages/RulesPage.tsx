import { useMemo, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router";
import { EmptyState, ErrorPanel, PageHeader, Spinner } from "../components/States";
import {
  ApiError,
  createRule,
  deleteRule,
  fetchCameras,
  fetchRules,
  fetchZones,
  updateRule,
} from "../lib/api";
import { useAuth } from "../lib/auth";
import { severityClass } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";
import type { RuleCreate, RuleOut } from "../lib/types";

const SUPPORTED_RULE_TYPES = [
  "zone_enter",
  "zone_dwell",
  "line_cross",
  "restricted_zone_intrusion",
] as const;

const EVENT_TYPES = [
  "zone_enter",
  "dwell_violation",
  "line_crossing",
  "restricted_zone_intrusion",
  "hazardous_zone_intrusion",
  "after_hours_movement",
];

interface RuleDraft {
  rule_id: string;
  name: string;
  rule_type: string;
  event_type: string;
  severity: string;
  enabled: boolean;
  camera_id: string;
  zone_ids: string;
  params_json: string;
  cooldown_seconds: string;
  confirm_seconds: string;
  min_confidence: string;
  line: string;
}

function emptyDraft(): RuleDraft {
  return {
    rule_id: "",
    name: "",
    rule_type: "restricted_zone_intrusion",
    event_type: "restricted_zone_intrusion",
    severity: "high",
    enabled: true,
    camera_id: "",
    zone_ids: "",
    params_json: "{}",
    cooldown_seconds: "60",
    confirm_seconds: "2",
    min_confidence: "0.4",
    line: "",
  };
}

function fromRule(rule: RuleOut): RuleDraft {
  return {
    rule_id: rule.rule_id,
    name: rule.name,
    rule_type: rule.rule_type,
    event_type: rule.event_type,
    severity: rule.severity,
    enabled: rule.enabled,
    camera_id: rule.camera_id ?? "",
    zone_ids: rule.zone_ids.join(", "),
    params_json: JSON.stringify(rule.params ?? {}, null, 0),
    cooldown_seconds: String(rule.cooldown_seconds),
    confirm_seconds: String(rule.confirm_seconds),
    min_confidence: String(rule.min_confidence),
    line: (rule.line ?? []).map((p) => p.join(",")).join(" "),
  };
}

function parseLine(raw: string): number[][] | undefined {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  return trimmed.split(/\s+/).map((pair) => {
    const [x, y] = pair.split(",").map((n) => Number(n.trim()));
    if (Number.isNaN(x) || Number.isNaN(y)) {
      throw new Error(`invalid line point '${pair}' — use x,y pairs`);
    }
    if (!(x >= 0 && x <= 1 && y >= 0 && y <= 1)) {
      throw new Error(
        `line point '${pair}' is out of range — coordinates must be normalized 0–1`,
      );
    }
    return [x, y];
  });
}

export default function RulesPage() {
  const { can } = useAuth();
  const { tick } = useLive();
  const [searchParams, setSearchParams] = useSearchParams();
  const cameraFilter = searchParams.get("camera_id") ?? "";
  const enabledFilter = searchParams.get("enabled") ?? "";

  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState<RuleDraft>(emptyDraft);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<ApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const rules = useQuery(
    (signal) =>
      fetchRules(
        {
          camera_id: cameraFilter || undefined,
          enabled: enabledFilter ? enabledFilter === "true" : undefined,
          limit: 200,
        },
        signal,
      ),
    [cameraFilter, enabledFilter, tick],
  );
  const cameras = useQuery((signal) => fetchCameras({ limit: 500 }, signal), []);
  const zones = useQuery(
    (signal) =>
      fetchZones(
        { camera_id: draft.camera_id || undefined, limit: 100 },
        signal,
      ),
    [draft.camera_id],
  );

  const canManage = can("rules:manage");
  const zoneOptions = useMemo(() => zones.data?.items ?? [], [zones.data]);

  function set<K extends keyof RuleDraft>(key: K, value: RuleDraft[K]) {
    setDraft((d) => ({ ...d, [key]: value }));
  }

  function startEdit(rule: RuleOut) {
    setEditingId(rule.rule_id);
    setDraft(fromRule(rule));
    setFormError(null);
    setNotice(null);
    setShowForm(true);
  }

  function resetForm() {
    setEditingId(null);
    setDraft(emptyDraft());
    setFormError(null);
    setNotice(null);
    setShowForm(false);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setFormError(null);
    setNotice(null);
    try {
      let params: Record<string, unknown>;
      try {
        params = JSON.parse(draft.params_json || "{}");
      } catch {
        throw new ApiError("invalid_params", "params must be valid JSON.", 422);
      }
      if (typeof params !== "object" || params === null || Array.isArray(params)) {
        throw new ApiError("invalid_params", "params must be a JSON object.", 422);
      }
      let line: number[][] | undefined;
      try {
        line = parseLine(draft.line);
      } catch (err) {
        throw new ApiError("invalid_line", String(err instanceof Error ? err.message : err), 422);
      }
      const zoneIds = draft.zone_ids
        .split(",")
        .map((z) => z.trim())
        .filter(Boolean);

      const editing = editingId;
      if (editing) {
        await updateRule(editing, {
          name: draft.name,
          event_type: draft.event_type,
          severity: draft.severity as RuleCreate["severity"],
          enabled: draft.enabled,
          zone_ids: zoneIds,
          params: params as Record<string, unknown>,
          cooldown_seconds: Number(draft.cooldown_seconds),
          confirm_seconds: Number(draft.confirm_seconds),
          min_confidence: Number(draft.min_confidence),
          line: line ?? null,
        });
      } else {
        await createRule({
          rule_id: draft.rule_id.trim(),
          name: draft.name.trim(),
          rule_type: draft.rule_type,
          event_type: draft.event_type,
          severity: draft.severity as RuleCreate["severity"],
          enabled: draft.enabled,
          zone_ids: zoneIds,
          params: params as Record<string, unknown>,
          cooldown_seconds: Number(draft.cooldown_seconds),
          confirm_seconds: Number(draft.confirm_seconds),
          min_confidence: Number(draft.min_confidence),
          camera_id: draft.camera_id || null,
          line: line ?? null,
        });
      }
      resetForm();
      setNotice(
        editing
          ? `Rule ${editing} updated (server bumps its version).`
          : "Rule created.",
      );
      rules.reload();
    } catch (err) {
      setFormError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  async function remove(rule: RuleOut) {
    try {
      await deleteRule(rule.rule_id);
      if (editingId === rule.rule_id) resetForm();
      rules.reload();
    } catch (err) {
      setFormError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    }
  }

  return (
    <div>
      <PageHeader
        title="Rule Configuration"
        subtitle="Detection rules with cooldowns, confidence thresholds and schedules."
        actions={
          canManage ? (
            <button
              type="button"
              onClick={() => (showForm ? resetForm() : setShowForm(true))}
              className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white"
              data-testid="add-rule"
            >
              {showForm ? "Close form" : "New rule"}
            </button>
          ) : null
        }
      />

      <div className="mb-4 flex flex-wrap gap-3">
        <label className="text-xs text-slate-400">
          Camera{" "}
          <select
            value={cameraFilter}
            onChange={(e) => {
              const params = new URLSearchParams(searchParams);
              if (e.target.value) params.set("camera_id", e.target.value);
              else params.delete("camera_id");
              setSearchParams(params);
            }}
            className="ml-1 rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100"
            data-testid="rule-camera-filter"
          >
            <option value="">All</option>
            {(cameras.data?.items ?? [])
              .filter((c) => !c.deleted_at)
              .map((c) => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.name}
                </option>
              ))}
          </select>
        </label>
        <label className="text-xs text-slate-400">
          State{" "}
          <select
            value={enabledFilter}
            onChange={(e) => {
              const params = new URLSearchParams(searchParams);
              if (e.target.value) params.set("enabled", e.target.value);
              else params.delete("enabled");
              setSearchParams(params);
            }}
            className="ml-1 rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-sm text-slate-100"
          >
            <option value="">All</option>
            <option value="true">Enabled</option>
            <option value="false">Disabled</option>
          </select>
        </label>
      </div>

      {showForm && canManage ? (
        <form
          onSubmit={submit}
          className="mb-5 rounded-xl border border-slate-800 bg-slate-900/60 p-4"
          data-testid="rule-form"
        >
          <h2 className="text-sm font-semibold text-slate-200">
            {editingId ? `Edit rule · ${editingId}` : "New rule"}
          </h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <label className="block text-xs text-slate-400">
              Rule ID {editingId ? "" : "(required, [a-zA-Z0-9_-]+)"}
              <input
                value={draft.rule_id}
                onChange={(e) => set("rule_id", e.target.value)}
                disabled={Boolean(editingId)}
                required={!editingId}
                pattern="[a-zA-Z0-9_\-]+"
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100 disabled:text-slate-500"
                data-testid="rule-form-id"
              />
            </label>
            <label className="block text-xs text-slate-400">
              Name (required)
              <input
                value={draft.name}
                onChange={(e) => set("name", e.target.value)}
                required
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                data-testid="rule-form-name"
              />
            </label>
            <label className="block text-xs text-slate-400">
              Rule type
              <select
                value={draft.rule_type}
                onChange={(e) => set("rule_type", e.target.value)}
                disabled={Boolean(editingId)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                data-testid="rule-form-type"
              >
                {SUPPORTED_RULE_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-slate-400">
              Event type
              <select
                value={draft.event_type}
                onChange={(e) => set("event_type", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              >
                {!EVENT_TYPES.includes(draft.event_type as (typeof EVENT_TYPES)[number]) ? (
                  <option value={draft.event_type}>{draft.event_type}</option>
                ) : null}
                {EVENT_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-slate-400">
              Severity
              <select
                value={draft.severity}
                onChange={(e) => set("severity", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              >
                {["low", "medium", "high", "critical"].map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-slate-400">
              Camera (scope, optional)
              <select
                value={draft.camera_id}
                onChange={(e) => set("camera_id", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                data-testid="rule-form-camera"
              >
                <option value="">All cameras</option>
                {(cameras.data?.items ?? [])
                  .filter((c) => !c.deleted_at)
                  .map((c) => (
                    <option key={c.camera_id} value={c.camera_id}>
                      {c.name}
                    </option>
                  ))}
              </select>
            </label>
            <label className="block text-xs text-slate-400">
              Zone IDs (comma separated)
              <input
                value={draft.zone_ids}
                onChange={(e) => set("zone_ids", e.target.value)}
                placeholder="z1, z2"
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                data-testid="rule-form-zones"
              />
              {zoneOptions.length > 0 ? (
                <span className="mt-1 block text-[10px] text-slate-600">
                  available: {zoneOptions.map((z) => z.name).join(", ")}
                </span>
              ) : null}
            </label>
            <label className="block text-xs text-slate-400">
              Cooldown (s)
              <input
                type="number"
                min={0}
                value={draft.cooldown_seconds}
                onChange={(e) => set("cooldown_seconds", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              />
            </label>
            <label className="block text-xs text-slate-400">
              Confirm (s)
              <input
                type="number"
                min={0}
                value={draft.confirm_seconds}
                onChange={(e) => set("confirm_seconds", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              />
            </label>
            <label className="block text-xs text-slate-400">
              Min confidence (0–1)
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={draft.min_confidence}
                onChange={(e) => set("min_confidence", e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              />
            </label>
            <label className="block text-xs text-slate-400 sm:col-span-2">
              Line points (line_cross only): x,y x,y … — normalized 0–1
              <input
                value={draft.line}
                onChange={(e) => set("line", e.target.value)}
                placeholder="0.2,0.5 0.8,0.5"
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              />
            </label>
            <label className="block text-xs text-slate-400">
              Params (JSON)
              <textarea
                value={draft.params_json}
                onChange={(e) => set("params_json", e.target.value)}
                rows={2}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 font-mono text-xs text-slate-100"
                data-testid="rule-form-params"
              />
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={draft.enabled}
                onChange={(e) => set("enabled", e.target.checked)}
                className="size-4 accent-emerald-500"
              />
              Enabled
            </label>
          </div>
          {formError ? (
            <div className="mt-3">
              <ErrorPanel error={formError} />
            </div>
          ) : null}
          <div className="mt-3 flex gap-2">
            <button
              type="submit"
              disabled={busy}
              className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white disabled:opacity-50"
              data-testid="rule-form-submit"
            >
              {busy ? "Saving…" : editingId ? "Save changes" : "Create rule"}
            </button>
            <button
              type="button"
              onClick={resetForm}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900"
            >
              Cancel
            </button>
          </div>
        </form>
      ) : null}

      {notice ? (
        <p className="mb-4 text-sm text-emerald-400" role="status">
          {notice}
        </p>
      ) : null}

      {rules.error ? <ErrorPanel error={rules.error} onRetry={rules.reload} /> : null}
      {rules.loading ? <Spinner label="Loading rules" /> : null}
      {rules.data && rules.data.items.length === 0 && !rules.loading ? (
        <EmptyState title="No rules match" hint="Create a rule or clear filters." />
      ) : null}

      <ul className="space-y-2" data-testid="rules-list">
        {(rules.data?.items ?? []).map((rule) => (
          <li
            key={rule.id}
            className="rounded-xl border border-slate-800 bg-slate-900/60 p-3"
            data-testid={`rule-${rule.rule_id}`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${severityClass(rule.severity)}`}
              >
                {rule.severity}
              </span>
              <span className="text-sm font-medium text-slate-200">{rule.name}</span>
              <span className="font-mono text-xs text-slate-500">{rule.rule_id}</span>
              <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">
                {rule.rule_type}
              </span>
              {rule.enabled ? (
                <span className="text-[10px] font-semibold uppercase text-emerald-400">
                  enabled
                </span>
              ) : (
                <span className="text-[10px] font-semibold uppercase text-slate-500">
                  disabled
                </span>
              )}
              <span className="ml-auto text-xs text-slate-500">
                v{rule.version}
                {rule.camera_id ? ` · ${rule.camera_id}` : " · all cameras"}
              </span>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
              <span>event: {rule.event_type}</span>
              <span>cooldown: {rule.cooldown_seconds}s</span>
              <span>confirm: {rule.confirm_seconds}s</span>
              <span>min conf: {rule.min_confidence}</span>
              <span>
                zones: {rule.zone_ids.length ? rule.zone_ids.join(", ") : "none"}
              </span>
              <span>
                schedule:{" "}
                {rule.schedule.windows.length === 0
                  ? "always"
                  : `${rule.schedule.windows.length} window(s)`}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-slate-800 pt-2">
              <Link
                to={`/events?rule_id=${encodeURIComponent(rule.rule_id)}`}
                className="text-xs text-sky-400 hover:text-sky-300"
              >
                Events from this rule →
              </Link>
              {canManage ? (
                <>
                  <button
                    type="button"
                    onClick={() => startEdit(rule)}
                    className="rounded px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                    data-testid={`rule-edit-${rule.rule_id}`}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    onClick={() => void remove(rule)}
                    className="rounded px-2 py-1 text-xs text-rose-400 hover:bg-slate-800"
                    data-testid={`rule-delete-${rule.rule_id}`}
                  >
                    Delete
                  </button>
                </>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
