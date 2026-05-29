"use client";

import { AlertTriangle, Trash2, X } from "lucide-react";

type DeleteConfirmDialogProps = {
  open: boolean;
  title: string;
  itemName: string;
  description: string;
  details?: string[];
  confirmLabel: string;
  isDeleting?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

export function DeleteConfirmDialog({
  open,
  title,
  itemName,
  description,
  details = [],
  confirmLabel,
  isDeleting = false,
  onCancel,
  onConfirm
}: DeleteConfirmDialogProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-4 py-6" role="presentation">
      <section
        aria-modal="true"
        role="dialog"
        aria-labelledby="delete-dialog-title"
        className="w-full max-w-md rounded-lg border border-red-100 bg-white p-5 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex gap-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-red-50 text-red-700">
              <AlertTriangle size={20} />
            </span>
            <div>
              <h2 id="delete-dialog-title" className="text-lg font-bold text-ink">
                {title}
              </h2>
              <p className="mt-1 text-sm font-semibold text-red-700">{itemName}</p>
            </div>
          </div>
          <button className="icon-button h-8 w-8" type="button" aria-label="Close delete confirmation" onClick={onCancel}>
            <X size={16} />
          </button>
        </div>

        <p className="mt-4 text-sm leading-6 text-slate-600">{description}</p>

        {details.length ? (
          <div className="mt-4 rounded-md border border-red-100 bg-red-50 p-3">
            <p className="text-xs font-semibold uppercase text-red-800">This will remove</p>
            <ul className="mt-2 space-y-1 text-sm text-red-800">
              {details.map((detail) => (
                <li key={detail} className="flex gap-2">
                  <span aria-hidden="true">-</span>
                  <span>{detail}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <p className="mt-4 rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-600">This action cannot be undone.</p>

        <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button className="secondary-button" type="button" disabled={isDeleting} onClick={onCancel}>
            Cancel
          </button>
          <button
            className="inline-flex items-center justify-center gap-2 rounded-md bg-red-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-60"
            type="button"
            disabled={isDeleting}
            onClick={onConfirm}
          >
            <Trash2 size={16} />
            {isDeleting ? "Deleting..." : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
