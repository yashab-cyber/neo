from __future__ import annotations
from fastapi import APIRouter, Depends
from neo.cognitive import cognitive_settings
from typing import Any, Dict, List
from .auth import require_scope
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from neo.db import get_session, engine, Base
from neo.db.models import CognitiveRoadmapItem
from pydantic import BaseModel

router = APIRouter()


@router.get("/cognitive/config", dependencies=[Depends(require_scope("read:status"))])
async def cognitive_config():
    return cognitive_settings


ROADMAP_PHASES = {
    1: [
        "perception_layer.attention_mechanisms",
        "reasoning_engine.symbolic_reasoning",
        "executive_control.goal_management",
        "memory_systems.working_memory",
    ],
    # Phase 2 focuses on integration across subsystems: neuro-symbolic, advanced memory, meta-learning,
    # enhanced executive control, and unified attention. We map each high-level item to a concrete
    # configuration path present in the cognitive architecture config so automatic detection works.
    2: [
        # Neuro-symbolic integration layer
        "reasoning_engine.hybrid_integration.neuro_symbolic_bridges",
        # Advanced memory systems (episodic memory as representative advanced long-term capability)
        "memory_systems.long_term_memory.episodic_memory.storage_format",
        # Meta-learning capabilities
        "learning_systems.meta_learning.algorithms",
        # Enhanced executive control with goal management (priority scheduling)
        "executive_control.goal_management.priority_scheduling",
        # Attention mechanisms integrated (visual saliency maps)
        "perception_layer.attention_mechanisms.visual_attention.saliency_maps",
        # Linguistic self-attention (cross-modal integration piece)
        "perception_layer.attention_mechanisms.linguistic_attention.self_attention",
    ],
    # Phase 3: Advanced capabilities (distributed, continual learning, advanced reasoning, adaptation, monitoring)
    3: [
        # Distributed cognitive processing
        "distributed_processing.multi_agent.communication",
        # Continual learning framework
        "learning_systems.continual_learning.enabled",
        # Advanced reasoning (hybrid fusion already present, add symbolic grounding for differentiation)
        "reasoning_engine.hybrid_integration.symbolic_grounding",
        # Sophisticated adaptation mechanisms
        "adaptation.mechanisms.sophisticated",
        # Comprehensive performance monitoring metrics
        "performance_monitoring.metrics.enabled",
    ],
    # Phase 4: Optimization & scaling (efficiency, distributed scale, tuning, benchmarking, production readiness)
    4: [
        # Computational efficiency (parallel processing already in phase 2 domain but include memory optimization)
        "performance_optimization.memory_optimization.garbage_collection",
        # Scaling to distributed environments
        "distributed_processing.multi_agent.agents",
        # Fine-tuning cognitive parameters
        "tuning.parameters.auto_fine_tuning",
        # Advanced benchmarking (longitudinal analysis)
        "performance_monitoring.benchmarking.longitudinal",
        # Production readiness marker
        "deployment.readiness.status",
    ],
}

# Neural architecture component mapping (name -> list of config paths that must exist)
NEURAL_COMPONENTS = {
    "input_processing": ["neural_networks.input_processing.unified_representation"],
    "transformer": ["neural_networks.transformer.attention_mechanisms"],
    "convolutional": ["neural_networks.convolutional.feature_extraction"],
    "spiking": ["neural_networks.spiking.synaptic_plasticity"],
    "recursive": ["neural_networks.recursive.architecture_search"],
    "memory_augmented_ntm": ["neural_networks.memory_augmented.ntm"],
    "memory_augmented_dnc": ["neural_networks.memory_augmented.dnc"],
    "attention_multi_head": ["neural_networks.attention.multi_head"],
    "attention_cross_modal": ["neural_networks.attention.cross_modal"],
    "generative_vae": ["neural_networks.generative.vae"],
    "generative_gan": ["neural_networks.generative.gan"],
    "gnn": ["neural_networks.gnn.message_passing"],
    "capsule": ["neural_networks.capsule.dynamic_routing"],
    "training_progressive_growing": ["neural_networks.training_strategies.progressive_growing"],
    "training_transfer_learning": ["neural_networks.training_strategies.transfer_learning"],
}

# AI Engine architecture component domains mapping to config paths
AI_ENGINE_COMPONENTS = {
    "input_layer": [
        "ai_engine.input_layer.text_input",
        "ai_engine.input_layer.voice_input",
        "ai_engine.input_layer.sensor_input",
        "ai_engine.input_layer.context_input",
        "ai_engine.input_layer.historical_data",
    ],
    "preprocessing_pipeline": [
        "ai_engine.preprocessing.nlp",
        "ai_engine.preprocessing.asr",
        "ai_engine.preprocessing.cv",
        "ai_engine.preprocessing.data_normalization",
        "ai_engine.preprocessing.feature_extraction",
    ],
    "deep_learning_paradigm": [
        "ai_engine.learning_core.deep.convolutional_networks",
        "ai_engine.learning_core.deep.recurrent_networks",
        "ai_engine.learning_core.deep.transformer_networks",
        "ai_engine.learning_core.deep.generative_networks",
    ],
    "neuro_learning_paradigm": [
        "ai_engine.learning_core.neuro.spiking_networks",
        "ai_engine.learning_core.neuro.plasticity_learning",
        "ai_engine.learning_core.neuro.attention_networks",
        "ai_engine.learning_core.neuro.memory_networks",
    ],
    "recursive_learning_paradigm": [
        "ai_engine.learning_core.recursive.reinforcement_learning",
        "ai_engine.learning_core.recursive.meta_learning",
        "ai_engine.learning_core.recursive.adaptive_learning",
        "ai_engine.learning_core.recursive.self_improvement_logic",
    ],
    "integration_layer": [
        "ai_engine.integration.paradigm_coordinator",
        "ai_engine.integration.decision_fusion",
        "ai_engine.integration.context_engine",
        "ai_engine.integration.conflict_resolution",
    ],
    "output_generation": [
        "ai_engine.output_generation.natural_language_generation",
        "ai_engine.output_generation.action_commands",
        "ai_engine.output_generation.system_controls",
        "ai_engine.output_generation.visualizations",
    ],
    "feedback_learning": [
        "ai_engine.feedback.performance_monitor",
        "ai_engine.feedback.learning_updates",
        "ai_engine.feedback.weight_updates",
        "ai_engine.feedback.model_selection",
    ],
    "decision_fusion_details": [
        "ai_engine.decision_fusion.weight_assignment",
        "ai_engine.decision_fusion.confidence_scoring",
        "ai_engine.decision_fusion.ensemble_methods",
        "ai_engine.decision_fusion.voting_mechanisms",
        "ai_engine.decision_fusion.context.historical_context",
        "ai_engine.decision_fusion.context.current_context",
        "ai_engine.decision_fusion.context.predictive_context",
        "ai_engine.decision_fusion.context.user_context",
        "ai_engine.decision_fusion.final.decision_fusion",
        "ai_engine.decision_fusion.final.validation",
        "ai_engine.decision_fusion.final.execution",
        "ai_engine.decision_fusion.final.feedback_loop",
    ],
    "performance_monitoring": [
        "ai_engine.performance.monitoring.latency_monitoring",
        "ai_engine.performance.monitoring.throughput_tracking",
        "ai_engine.performance.monitoring.accuracy_measurement",
        "ai_engine.performance.monitoring.resource_usage",
        "ai_engine.performance.monitoring.error_detection",
    ],
    "optimization_engine": [
        "ai_engine.performance.optimization.auto_tuning",
        "ai_engine.performance.optimization.auto_scaling",
        "ai_engine.performance.optimization.load_balancing",
        "ai_engine.performance.optimization.caching_strategies",
        "ai_engine.performance.optimization.model_pruning",
    ],
    "adaptation_triggers": [
        "ai_engine.performance.adaptation.threshold_monitoring",
        "ai_engine.performance.adaptation.anomaly_detection",
        "ai_engine.performance.adaptation.trend_analysis",
        "ai_engine.performance.adaptation.performance_prediction",
    ],
}

# Monitoring & performance domains mapping (extracted from monitoring-performance documentation)
MONITORING_COMPONENTS = {
    "data_collection_layer": [
        "monitoring_system.data_collection.system_metrics.metrics",
        "monitoring_system.data_collection.application_metrics.metrics",
        "monitoring_system.data_collection.ai_metrics.metrics",
    ],
    "alerting_rules": [
        "monitoring_system.alerting.alert_rules.critical_alerts",
    ],
    "notification_channels": [
        "monitoring_system.alerting.notification_channels.email",
        "monitoring_system.alerting.notification_channels.slack",
        "monitoring_system.alerting.notification_channels.pagerduty",
    ],
    "dashboards_core": [
        "monitoring_system.dashboards.system_overview.panels",
        "monitoring_system.dashboards.ai_performance.panels",
    ],
    "storage_backends": [
        "monitoring_system.storage.time_series.engine",
        "monitoring_system.storage.logs.engine",
        "monitoring_system.storage.traces.engine",
    ],
    "performance_optimization": [
        "monitoring_system.performance_optimization.auto_scaling.enabled",
        "monitoring_system.performance_optimization.caching.enabled",
    ],
    "resource_optimization": [
        "monitoring_system.performance_optimization.resource_optimization.enabled",
    ],
    "compliance_security": [
        "compliance.security.encryption_at_rest",
        "compliance.security.encryption_in_transit",
        "compliance.security.access_control",
    ],
    "reporting": [
        "compliance.reporting.sla_reporting",
        "compliance.reporting.performance_reports",
        "compliance.reporting.capacity_reports",
    ],
}

# Security framework domains mapping (from security-framework documentation)
SECURITY_FRAMEWORK_COMPONENTS = {
    "perimeter_defense": [
        "security_framework.perimeter.firewall_systems",
        "security_framework.perimeter.intrusion_detection",
        "security_framework.perimeter.intrusion_prevention",
        "security_framework.perimeter.web_application_firewall",
    ],
    "network_security": [
        "security_framework.network.vpn_gateways",
        "security_framework.network.network_segmentation",
        "security_framework.network.zero_trust_network",
    ],
    "application_security": [
        "security_framework.application.authentication_systems",
        "security_framework.application.authorization_controls",
        "security_framework.application.multi_factor_auth",
        "security_framework.application.role_based_access",
    ],
    "data_security": [
        "security_framework.data.encryption_at_rest",
        "security_framework.data.transport_encryption",
        "security_framework.data.data_loss_prevention",
        "security_framework.data.secure_backup",
        "security_framework.data.key_management",
    ],
    "ai_security": [
        "security_framework.ai.adversarial_defense",
        "security_framework.ai.model_security",
        "security_framework.ai.privacy_preservation",
        "security_framework.ai.ai_audit_trail",
        "security_framework.ai.bias_detection",
    ],
    "system_security": [
        "security_framework.system.hardware_security_module",
        "security_framework.system.trusted_platform_module",
        "security_framework.system.secure_boot",
        "security_framework.system.system_integrity",
        "security_framework.system.sandboxing",
    ],
    "monitoring_response": [
        "security_framework.monitoring.siem",
        "security_framework.monitoring.soc",
        "security_framework.monitoring.incident_response",
        "security_framework.monitoring.digital_forensics",
        "security_framework.monitoring.threat_intelligence",
    ],
    "identity_access_management": [
        "security_framework.iam.identity_sources",
        "security_framework.iam.authentication_layer",
        "security_framework.iam.multi_factor_authentication",
        "security_framework.iam.authorization_engine",
        "security_framework.iam.session_management",
    ],
    "threat_detection_response": [
        "security_framework.threat_detection.data_collection",
        "security_framework.threat_detection.analysis_engine",
        "security_framework.threat_detection.threat_intelligence",
        "security_framework.threat_detection.response_actions",
        "security_framework.threat_detection.orchestration",
    ],
    "cryptographic_architecture": [
        "security_framework.crypto.key_management",
        "security_framework.crypto.encryption_services",
        "security_framework.crypto.digital_signatures",
        "security_framework.crypto.secure_communications",
    ],
    "compliance_governance": [
        "security_framework.governance.regulatory_frameworks",
        "security_framework.governance.policy_management",
        "security_framework.governance.audit_reporting",
        "security_framework.governance.training_awareness",
    ],
}

# System architecture overview domains (from system-overview documentation)
SYSTEM_ARCH_COMPONENTS = {
    "user_interface_layer": ["system_overview.user_interface.ui", "system_overview.user_interface.voice", "system_overview.user_interface.cli", "system_overview.user_interface.rest_api"],
    "interaction_processing": ["system_overview.interaction_processing.nlp", "system_overview.interaction_processing.command_parser", "system_overview.interaction_processing.context_manager", "system_overview.interaction_processing.session_manager"],
    "ai_engine_core": ["system_overview.ai_engine.multi_paradigm.deep_learning", "system_overview.ai_engine.multi_paradigm.neuro_learning", "system_overview.ai_engine.multi_paradigm.recursive_learning", "system_overview.ai_engine.decision.smart_thinking_framework"],
    "knowledge_management": ["system_overview.ai_engine.knowledge.knowledge_graph", "system_overview.ai_engine.knowledge.memory_system", "system_overview.ai_engine.knowledge.learning_engine", "system_overview.ai_engine.knowledge.adaptation_engine"],
    "security_layer": ["system_overview.security.threat_detection", "system_overview.security.security_orchestrator", "system_overview.security.incident_response", "system_overview.security.encryption_engine"],
    "system_intelligence": ["system_overview.system_intelligence.process_management", "system_overview.system_intelligence.file_management", "system_overview.system_intelligence.network_management", "system_overview.system_intelligence.automation_engine"],
    "data_management": ["system_overview.data.data_storage", "system_overview.data.database_management", "system_overview.data.backup_system", "system_overview.data.data_pipeline"],
    "integration_layer": ["system_overview.integration.cloud_connector", "system_overview.integration.third_party_api", "system_overview.integration.device_interface", "system_overview.integration.event_bus"],
    "infrastructure_layer": ["system_overview.infrastructure.container_runtime", "system_overview.infrastructure.orchestration", "system_overview.infrastructure.monitoring", "system_overview.infrastructure.logging"],
    "performance_scalability": ["system_overview.performance.response_time", "system_overview.performance.throughput", "system_overview.performance.auto_scaling"],
    "deployment_architecture": ["system_overview.deployment.kubernetes_cluster", "system_overview.deployment.service_discovery", "system_overview.deployment.load_balancing"],
    "quality_attributes": ["system_overview.quality.performance", "system_overview.quality.reliability", "system_overview.quality.security", "system_overview.quality.scalability"],
}


class RoadmapUpdate(BaseModel):
    implemented: bool
    description: str | None = None


async def _auto_create():  # pragma: no cover
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@router.get("/cognitive/roadmap", dependencies=[Depends(require_scope("read:status"))])
async def cognitive_roadmap(session: AsyncSession = Depends(get_session)):
    await _auto_create()
    db_items = { (r.phase, r.item): r for r in (await session.execute(select(CognitiveRoadmapItem))).scalars() }

    def has(path: str) -> bool:
        cur: Any = cognitive_settings.get("cognitive_architecture", {})
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        return True

    phase_status: Dict[int, Dict[str, Any]] = {}
    for phase, items in ROADMAP_PHASES.items():
        statuses: List[Dict[str, Any]] = []
        for item in items:
            key = (phase, item)
            if key in db_items:
                rec = db_items[key]
                statuses.append({"item": item, "implemented": rec.implemented, "description": rec.description})
            else:
                statuses.append({"item": item, "implemented": has(item), "description": None})
        phase_status[phase] = {
            "complete": all(s["implemented"] for s in statuses),
            "items": statuses,
        }
    return {"phases": phase_status}


@router.get("/cognitive/integration/status", dependencies=[Depends(require_scope("read:status"))])
async def cognitive_integration_status():
    """Return summarized status for Phase 2 integration domains.

    Derived heuristically from config presence. This gives a quick roll-up separate from
    the fine-grained roadmap list. Each domain returns implemented True if its
    representative config path(s) are present.
    """
    cfg = cognitive_settings.get("cognitive_architecture", {})

    def has(path: str) -> bool:
        cur: Any = cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        return True

    domains = {
        "neuro_symbolic": has("reasoning_engine.hybrid_integration.neuro_symbolic_bridges"),
        "advanced_memory": has("memory_systems.long_term_memory.episodic_memory") and has("memory_systems.long_term_memory.semantic_memory"),
        "meta_learning": has("learning_systems.meta_learning.algorithms"),
        "executive_control": has("executive_control.goal_management.priority_scheduling"),
        "attention_integration": has("perception_layer.attention_mechanisms.visual_attention.saliency_maps") and has("perception_layer.attention_mechanisms.linguistic_attention.self_attention"),
    }
    complete = all(domains.values())
    return {"domains": domains, "complete": complete}


@router.get("/cognitive/advanced/status", dependencies=[Depends(require_scope("read:status"))])
async def cognitive_advanced_status():
    """Roll-up status for phases 3 & 4 advanced and optimization domains."""
    cfg = cognitive_settings.get("cognitive_architecture", {})

    def has(path: str) -> bool:
        cur: Any = cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return False
            cur = cur[part]
        return True

    domains = {
        "distributed_processing": has("distributed_processing.multi_agent.communication") and has("distributed_processing.multi_agent.agents"),
        "continual_learning": has("learning_systems.continual_learning.enabled"),
        "advanced_reasoning": has("reasoning_engine.hybrid_integration.symbolic_grounding"),
        "adaptation_mechanisms": has("adaptation.mechanisms.sophisticated"),
        "performance_monitoring": has("performance_monitoring.metrics.enabled") and has("performance_monitoring.benchmarking.longitudinal"),
        "optimization_scaling": has("performance_optimization.memory_optimization.garbage_collection") and has("tuning.parameters.auto_fine_tuning"),
        "production_readiness": has("deployment.readiness.status"),
    }
    complete = all(domains.values())
    return {"domains": domains, "complete": complete}


def _cfg_has(path: str) -> bool:
    cfg = cognitive_settings.get("cognitive_architecture", {})
    cur: Any = cfg
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


@router.get("/cognitive/neural/overview", dependencies=[Depends(require_scope("read:status"))])
async def neural_architecture_overview():
    """Report detection status for neural network architecture components.

    Implementation is heuristic: a component is considered implemented if all required
    config paths exist in the cognitive configuration. Returns summary counts and list.
    """
    components = []
    implemented_count = 0
    for name, paths in NEURAL_COMPONENTS.items():
        ok = all(_cfg_has(p) for p in paths)
        if ok:
            implemented_count += 1
        components.append({"component": name, "implemented": ok})
    total = len(NEURAL_COMPONENTS)
    return {"total": total, "implemented": implemented_count, "completion": implemented_count / total if total else 0.0, "components": components}


@router.get("/cognitive/ai-engine/status", dependencies=[Depends(require_scope("read:status"))])
async def ai_engine_status():
    """Report AI Engine architecture domain completion status.

    A domain is complete if all its mapped config paths exist. Provides per-domain results
    and overall completion percentage.
    """
    domain_results = []
    full = 0
    for domain, paths in AI_ENGINE_COMPONENTS.items():
        ok = all(_cfg_has(p) for p in paths)
        if ok:
            full += 1
        domain_results.append({"domain": domain, "implemented": ok, "paths": paths})
    total = len(AI_ENGINE_COMPONENTS)
    return {
        "domains": domain_results,
        "implemented": full,
        "total": total,
        "completion": full / total if total else 0.0,
    }


@router.get("/cognitive/monitoring/status", dependencies=[Depends(require_scope("read:status"))])
async def monitoring_status():
    """Return status for monitoring & performance domains."""
    results = []
    implemented = 0
    for name, paths in MONITORING_COMPONENTS.items():
        ok = all(_cfg_has(p) for p in paths)
        if ok:
            implemented += 1
        results.append({"domain": name, "implemented": ok, "paths": paths})
    total = len(MONITORING_COMPONENTS)
    return {"domains": results, "implemented": implemented, "total": total, "completion": implemented / total if total else 0.0}


@router.get("/cognitive/security/status", dependencies=[Depends(require_scope("read:status"))])
async def security_framework_status():
    """Return status for security framework architecture domains."""
    domains = []
    complete = 0
    for name, paths in SECURITY_FRAMEWORK_COMPONENTS.items():
        ok = all(_cfg_has(p) for p in paths)
        if ok:
            complete += 1
        domains.append({"domain": name, "implemented": ok, "paths": paths})
    total = len(SECURITY_FRAMEWORK_COMPONENTS)
    return {"domains": domains, "implemented": complete, "total": total, "completion": complete / total if total else 0.0}


@router.get("/cognitive/system/status", dependencies=[Depends(require_scope("read:status"))])
async def system_architecture_status():
    """Return status for high-level system architecture domains."""
    results = []
    ok_count = 0
    for name, paths in SYSTEM_ARCH_COMPONENTS.items():
        ok = all(_cfg_has(p) for p in paths)
        if ok:
            ok_count += 1
        results.append({"domain": name, "implemented": ok, "paths": paths})
    total = len(SYSTEM_ARCH_COMPONENTS)
    return {"domains": results, "implemented": ok_count, "total": total, "completion": ok_count / total if total else 0.0}


@router.patch("/cognitive/roadmap/{phase}/{item}", dependencies=[Depends(require_scope("manage:cognitive"))])
async def update_roadmap_item(phase: int, item: str, body: RoadmapUpdate, session: AsyncSession = Depends(get_session)):
    await _auto_create()
    stmt = select(CognitiveRoadmapItem).where(CognitiveRoadmapItem.phase == phase, CognitiveRoadmapItem.item == item)
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing:
        existing.implemented = body.implemented
        existing.description = body.description
    else:
        session.add(CognitiveRoadmapItem(phase=phase, item=item, implemented=body.implemented, description=body.description))
    await session.commit()
    return {"phase": phase, "item": item, "implemented": body.implemented, "description": body.description}
