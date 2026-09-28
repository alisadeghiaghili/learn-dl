import { describeGoal } from '../levels/goalMatch';
import type { GoalSpec, Level } from '../levels/types';

export interface GoalPanelProps {
  level: Level | null;
  showSolution: boolean;
}

export function GoalPanel({ level, showSolution }: GoalPanelProps) {
  if (!level) {
    return (
      <div className="goal-panel">
        <h3>Sandbox</h3>
        <p className="muted">
          Free play. Type <code>levels</code> to learn, or build any architecture and train.
        </p>
      </div>
    );
  }

  const goal: GoalSpec = level.goal;
  return (
    <div className="goal-panel">
      <h3>Goal</h3>
      <pre className="goal-pre">{describeGoal(goal)}</pre>
      {showSolution && (
        <>
          <h3>Solution</h3>
          <pre className="goal-pre">{level.solutionCommand}</pre>
        </>
      )}
    </div>
  );
}
