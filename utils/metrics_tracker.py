"""Metrics tracking utilities for SUMO test/evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class MetricsTracker:
    """Track per-step SUMO metrics and finalize per-episode reports."""

    def __init__(self, model_name: str, scenario: Optional[str] = None, num_agents: int = 1) -> None:
        self.model_name = str(model_name)
        self.scenario = scenario
        self.num_agents = max(1, int(num_agents))
        self.report_payload: Optional[dict[str, Any]] = None
        self.reset_episode()

    def reset_episode(self) -> None:
        self.simulation_seconds = 0
        self.arrived_total = 0

        self.step_speed_samples: list[float] = []
        self.step_queue_samples: list[float] = []
        self.step_waiting_time_samples: list[float] = []
        self.step_time_loss_samples: list[float] = []
        self.step_green_samples: list[float] = []

        self.vehicle_delay_agg: dict[str, list[float]] = {}
        self.vehicle_seen_ids: set[str] = set()
        self.vehicle_stop_seconds: dict[str, int] = {}

        self.phase_switch_count = 0
        self.teleportation_by_collisions = 0
        self.fuel_total_litre = 0.0
        self.co2_total_mg = 0.0

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def episode_result_template() -> dict:
        return {
            "simulation_seconds": 0,
            "decision_steps": 0,
            "arrived_vehicles": 0,
            "safety_fallback_count": 0,
            "avg_delay_per_vehicle": 0.0,
            "avg_speed_incoming_road": 0.0,
            "avg_queue_length": 0.0,
            "max_queue_length": 0.0,
            "queue_length_variance": 0.0,
            "avg_waiting_time": 0.0,
            "time_loss": 0.0,
            "phase_switch_count": 0,
            "avg_stop_time_per_vehicle": 0.0,
            "teleportation_by_collisions": 0,
            "fuel_consumption_total_litre": 0.0,
            "intersection_throughput_veh_per_step": 0.0,
            "intersection_throughput_veh_per_hour": 0.0,
            "green_wave_effectiveness": 0.0,
            "co2_emissions_total_mg": 0.0,
        }

    def _collect_step_metrics(self, controlled_lanes: list[str]) -> dict:
        import traci

        lane_speeds: list[float] = []
        queue_length = 0.0
        lane_vehicle_ids: set[str] = set()

        for lane_id in controlled_lanes:
            try:
                lane_speeds.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
                queue_length += float(traci.lane.getLastStepHaltingNumber(lane_id))
                lane_vehicle_ids.update(traci.lane.getLastStepVehicleIDs(lane_id))
            except Exception:
                continue

        vehicle_delays: dict[str, float] = {}
        moving_count = 0
        waiting_time_samples: list[float] = []
        time_loss_samples: list[float] = []
        stopped_vehicle_ids: set[str] = set()

        for veh_id in lane_vehicle_ids:
            try:
                speed = float(traci.vehicle.getSpeed(veh_id))
                allowed_speed = float(traci.vehicle.getAllowedSpeed(veh_id))
                if speed > 0.1:
                    moving_count += 1
                else:
                    stopped_vehicle_ids.add(veh_id)
                if allowed_speed > 1e-6:
                    vehicle_delays[veh_id] = max(0.0, (allowed_speed - speed) / allowed_speed)
                else:
                    vehicle_delays[veh_id] = 0.0

                try:
                    waiting_time_samples.append(float(traci.vehicle.getAccumulatedWaitingTime(veh_id)))
                except Exception:
                    waiting_time_samples.append(0.0)
                try:
                    time_loss_samples.append(float(traci.vehicle.getTimeLoss(veh_id)))
                except Exception:
                    time_loss_samples.append(0.0)
            except Exception:
                vehicle_delays[veh_id] = 0.0

        co2_mg = 0.0
        fuel_mg = 0.0
        try:
            for veh_id in traci.vehicle.getIDList():
                co2_mg += float(traci.vehicle.getCO2Emission(veh_id))
                try:
                    fuel_mg += float(traci.vehicle.getFuelConsumption(veh_id))
                except Exception:
                    pass
        except Exception:
            pass

        teleport_count = 0
        collision_count = 0
        try:
            teleport_count = int(traci.simulation.getEndingTeleportNumber())
        except Exception:
            teleport_count = 0
        try:
            collision_count = int(traci.simulation.getCollidingVehiclesNumber())
        except Exception:
            collision_count = 0

        green_wave_effectiveness = 0.0
        if lane_vehicle_ids:
            green_wave_effectiveness = float(moving_count / len(lane_vehicle_ids))

        fuel_litre = float(fuel_mg / (1_000_000.0 * 0.745))

        return {
            "avg_lane_speed": self._mean(lane_speeds),
            "queue_length": queue_length,
            "vehicle_delays": vehicle_delays,
            "avg_waiting_time": self._mean(waiting_time_samples),
            "avg_time_loss": self._mean(time_loss_samples),
            "stopped_vehicle_ids": stopped_vehicle_ids,
            "lane_vehicle_ids": lane_vehicle_ids,
            "collision_teleport_count": int(min(teleport_count, collision_count)),
            "fuel_litre": fuel_litre,
            "green_wave_effectiveness": green_wave_effectiveness,
            "co2_mg": co2_mg,
        }

    def update_per_step(self, controlled_lanes: list[str]) -> None:
        metrics = self._collect_step_metrics(controlled_lanes)
        self.step_speed_samples.append(metrics["avg_lane_speed"])
        self.step_queue_samples.append(metrics["queue_length"])
        self.step_waiting_time_samples.append(metrics["avg_waiting_time"])
        self.step_time_loss_samples.append(metrics["avg_time_loss"])
        if metrics["vehicle_delays"]:
            self.step_green_samples.append(metrics["green_wave_effectiveness"])

        for vehicle_id, delay_value in metrics["vehicle_delays"].items():
            self.vehicle_delay_agg.setdefault(vehicle_id, []).append(delay_value)

        for vehicle_id in metrics["lane_vehicle_ids"]:
            self.vehicle_seen_ids.add(vehicle_id)
        for vehicle_id in metrics["stopped_vehicle_ids"]:
            self.vehicle_stop_seconds[vehicle_id] = int(self.vehicle_stop_seconds.get(vehicle_id, 0)) + 1

        self.teleportation_by_collisions += int(metrics["collision_teleport_count"])
        self.fuel_total_litre += float(metrics["fuel_litre"])
        self.co2_total_mg += float(metrics["co2_mg"])

        import traci

        self.arrived_total += int(traci.simulation.getArrivedNumber())
        self.simulation_seconds += 1

    def register_phase_switches(self, switch_count: int) -> None:
        self.phase_switch_count += int(switch_count)

    def end_episode(
        self,
        total_reward: float,
        arrived_vehicles: int,
        *,
        include_reward: bool = True,
        decision_steps: Optional[int] = None,
        safety_fallback_count: int = 0,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> dict:
        result = self.episode_result_template()
        result["simulation_seconds"] = int(self.simulation_seconds)
        if decision_steps is not None:
            result["decision_steps"] = int(decision_steps)

        result["arrived_vehicles"] = int(arrived_vehicles)
        result["safety_fallback_count"] = int(safety_fallback_count)
        result["avg_speed_incoming_road"] = self._mean(self.step_speed_samples)
        result["avg_queue_length"] = self._mean(self.step_queue_samples)
        result["max_queue_length"] = float(max(self.step_queue_samples)) if self.step_queue_samples else 0.0
        result["queue_length_variance"] = (
            float(np.std(np.asarray(self.step_queue_samples, dtype=np.float32)))
            if self.step_queue_samples
            else 0.0
        )
        result["avg_waiting_time"] = self._mean(self.step_waiting_time_samples)
        result["time_loss"] = self._mean(self.step_time_loss_samples)
        result["phase_switch_count"] = float(self.phase_switch_count / self.num_agents)

        if self.vehicle_seen_ids:
            stopped_seconds_total = float(
                sum(int(self.vehicle_stop_seconds.get(vehicle_id, 0)) for vehicle_id in self.vehicle_seen_ids)
            )
            result["avg_stop_time_per_vehicle"] = float(stopped_seconds_total / len(self.vehicle_seen_ids))
        else:
            result["avg_stop_time_per_vehicle"] = 0.0

        result["teleportation_by_collisions"] = int(self.teleportation_by_collisions)
        result["fuel_consumption_total_litre"] = float(self.fuel_total_litre)
        result["green_wave_effectiveness"] = self._mean(self.step_green_samples)
        result["co2_emissions_total_mg"] = float(self.co2_total_mg)

        per_vehicle_delay_means: list[float] = []
        for delay_samples in self.vehicle_delay_agg.values():
            if delay_samples:
                per_vehicle_delay_means.append(self._mean(delay_samples))
        result["avg_delay_per_vehicle"] = self._mean(per_vehicle_delay_means)

        if self.simulation_seconds > 0:
            throughput_per_second = float(arrived_vehicles / self.simulation_seconds)
            result["intersection_throughput_veh_per_step"] = throughput_per_second
            result["intersection_throughput_veh_per_hour"] = float(throughput_per_second * 3600.0)

        if include_reward:
            result["reward"] = float(total_reward)
        if extra_fields:
            result.update(extra_fields)

        return result

    @classmethod
    def aggregate_episode_results(cls, episode_results: list[dict]) -> dict:
        if not episode_results:
            return cls.episode_result_template()

        aggregated = cls.episode_result_template()
        aggregated["simulation_seconds"] = int(sum(item["simulation_seconds"] for item in episode_results))
        aggregated["decision_steps"] = int(sum(int(item.get("decision_steps", 0)) for item in episode_results))
        aggregated["arrived_vehicles"] = int(sum(item["arrived_vehicles"] for item in episode_results))
        aggregated["safety_fallback_count"] = int(
            sum(int(item.get("safety_fallback_count", 0)) for item in episode_results)
        )
        aggregated["phase_switch_count"] = cls._mean(
            [float(item.get("phase_switch_count", 0.0)) for item in episode_results]
        )
        aggregated["teleportation_by_collisions"] = int(
            sum(int(item.get("teleportation_by_collisions", 0)) for item in episode_results)
        )
        aggregated["fuel_consumption_total_litre"] = float(
            sum(float(item.get("fuel_consumption_total_litre", 0.0)) for item in episode_results)
        )
        aggregated["max_queue_length"] = float(
            max(float(item.get("max_queue_length", 0.0)) for item in episode_results)
        )

        scalar_keys = [
            "avg_delay_per_vehicle",
            "avg_speed_incoming_road",
            "avg_queue_length",
            "queue_length_variance",
            "avg_waiting_time",
            "time_loss",
            "avg_stop_time_per_vehicle",
            "intersection_throughput_veh_per_step",
            "intersection_throughput_veh_per_hour",
            "green_wave_effectiveness",
            "co2_emissions_total_mg",
        ]
        for key in scalar_keys:
            aggregated[key] = cls._mean([float(item[key]) for item in episode_results])

        return aggregated

    def save_results_to_json(self, filepath: str | Path) -> str:
        if self.report_payload is None:
            raise ValueError("report_payload is not set on MetricsTracker.")

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.report_payload, file, indent=2)
        return str(output_path)
