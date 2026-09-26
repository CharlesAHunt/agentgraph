<script lang="ts">
  import OptionPicker from "./OptionPicker.svelte";
  import { chat } from "./chat.svelte";
  import { CUSTOM_ROLE, MAX_ROLE_LENGTH, ROLES } from "./roles";

  const options = [...ROLES, { id: CUSTOM_ROLE, name: "Custom", summary: "Describe your own role" }];
</script>

<OptionPicker
  label="Role"
  legend="How the assistant should answer"
  value={chat.roleId}
  display={chat.role.name}
  {options}
  onselect={(id) => chat.setRole(id)}
  note="Shapes tone and focus. Answers still search and cite the corpus."
>
  {#snippet extra()}
    {#if chat.roleId === CUSTOM_ROLE}
      <div class="custom">
        <label class="sr-only" for="custom-role">Custom role</label>
        <textarea
          id="custom-role"
          rows="3"
          maxlength={MAX_ROLE_LENGTH}
          placeholder="e.g. Answer as a stellarator physicist and compare with tokamaks where relevant."
          value={chat.customRole}
          oninput={(e) => chat.setRole(CUSTOM_ROLE, e.currentTarget.value)}
        ></textarea>
        <span class="count">{chat.customRole.length} / {MAX_ROLE_LENGTH}</span>
      </div>
    {/if}
  {/snippet}
</OptionPicker>

<style>
  .custom {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 14px 8px 38px;
  }
  textarea {
    resize: vertical;
    min-height: 64px;
    padding: 8px 10px;
    border: 1px solid var(--rule);
    background: var(--surface-raised);
    font-size: 0.85rem;
    line-height: 1.45;
  }
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .count {
    align-self: flex-end;
    font-family: var(--font-mono);
    font-size: 0.64rem;
    color: var(--ink-faint);
    font-variant-numeric: tabular-nums;
  }
</style>
