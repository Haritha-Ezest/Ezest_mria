"""
Neo4j graph database client for medical knowledge graph operations.

This module provides the core Neo4j integration for building, querying, and managing
medical knowledge graphs with support for complex medical relationships and temporal data.
"""

from datetime import datetime
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncTransaction
from neo4j.exceptions import ClientError

from app.schemas.graph import (
    PatientGraphRequest, PatientGraphResponse,
    GraphQueryRequest, GraphQueryResponse, GraphStatistics, GraphHealthCheck,
    RelationshipType
)
from app.common.utils import get_logger


logger = get_logger(__name__)


class Neo4jGraphClient:
    """
    Neo4j client for medical knowledge graph operations.
    
    Handles connections, transactions, and medical-specific graph operations
    with proper error handling and retry logic.
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        max_connection_lifetime: Optional[int] = None,
        max_connection_pool_size: Optional[int] = None,
        connection_acquisition_timeout: Optional[int] = None
    ):
        """
        Initialize Neo4j client with connection parameters.
        
        Parameters default to configuration from environment variables.
        Explicit parameters override configuration defaults.
        
        Args:
            uri: Neo4j database URI (defaults to NEO4J_URI env var)
            username: Database username (defaults to NEO4J_USERNAME env var)
            password: Database password (defaults to NEO4J_PASSWORD env var)
            database: Database name (defaults to NEO4J_DATABASE env var)
            max_connection_lifetime: Max connection lifetime in seconds
            max_connection_pool_size: Maximum number of connections in pool
            connection_acquisition_timeout: Timeout for acquiring connections
        """
        from app.config import get_settings
        settings = get_settings()
        
        # Use provided parameters or fall back to configuration
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.max_connection_lifetime = max_connection_lifetime or settings.neo4j_max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size or settings.neo4j_max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout or settings.neo4j_connection_acquisition_timeout
        
        self.driver = None
        self._connection_config = {
            "max_connection_lifetime": self.max_connection_lifetime,
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_acquisition_timeout": self.connection_acquisition_timeout
        }
        
    def is_connected(self) -> bool:
        """Check if the client is connected to Neo4j."""
        return self.driver is not None
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database with robust error handling."""
        try:
            # Reload settings from config to ensure we have the latest credentials
            from app.config import get_settings
            settings = get_settings()
            
            # Always use the most up-to-date credentials
            self.uri = self.uri or settings.neo4j_uri
            self.username = self.username or settings.neo4j_username 
            self.password = self.password or settings.neo4j_password
            self.database = self.database or settings.neo4j_database
            
            # Log connection attempt (without sensitive info)
            logger.info(f"Connecting to Neo4j: {self.uri}, database: {self.database}")
            
            # Create driver with credentials
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                **self._connection_config
            )
            
            # Verify connectivity with timeout
            import asyncio
            try:
                await asyncio.wait_for(
                    self.driver.verify_connectivity(), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                raise Exception("Connection timeout while verifying Neo4j connectivity")
            
            # Initialize database schema
            await self._initialize_schema()
            
            logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            # Close driver if it was created but connection failed
            if self.driver:
                try:
                    await self.driver.close()
                except:
                    pass
                self.driver = None
            raise
    
    async def disconnect(self) -> None:
        """Close Neo4j database connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")    
    @asynccontextmanager
    async def session(self, **kwargs):
        """Async context manager for Neo4j sessions with robust reconnection handling."""
        # Try to ensure we have a valid connection
        retry_count = 0
        max_retries = 3
            
        while retry_count < max_retries:
            try:
                # If no driver or driver is closed, connect
                if not self.driver:
                    await self.connect()
                
                # Create a session
                session = self.driver.session(database=self.database, **kwargs)
                
                # Quick verification query
                if retry_count > 0:  # Only verify after a reconnection attempt
                    await session.run("RETURN 1")
                
                # If we get here, session is good
                try:
                    yield session
                finally:
                    await session.close()
                
                # If we complete successfully, exit the retry loop
                return
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to create valid Neo4j session after {max_retries} attempts. Last error: {e}")
                    raise
                
                logger.warning(f"Neo4j session error (attempt {retry_count}): {e}. Reconnecting...")
                
                # Close existing driver if any
                if self.driver:
                    try:
                        await self.driver.close()
                    except:
                        pass
                    self.driver = None
                      # Wait briefly before retry
                import asyncio
                await asyncio.sleep(0.5 * retry_count)
    
    def _convert_neo4j_types(self, obj):
        """
        Convert Neo4j types to Python-serializable types.
        
        This handles Neo4j-specific types like DateTime, Date, Time, Point, etc.
        that cannot be directly serialized by Pydantic/JSON.
        
        Args:
            obj: Object that may contain Neo4j types
            
        Returns:
            Object with Neo4j types converted to Python-native types
        """
        try:
            from neo4j.time import DateTime, Date, Time
            from neo4j.spatial import Point
            
            if obj is None:
                return None
            elif isinstance(obj, DateTime):
                # Convert Neo4j DateTime to ISO format string
                return obj.to_native().isoformat()
            elif isinstance(obj, Date):
                # Convert Neo4j Date to ISO format string
                return obj.to_native().isoformat()
            elif isinstance(obj, Time):
                # Convert Neo4j Time to ISO format string
                return obj.to_native().isoformat()
            elif isinstance(obj, Point):
                # Convert Neo4j Point to a dict with coordinates
                return {
                    "longitude": obj.longitude,
                    "latitude": obj.latitude,
                    "srid": obj.srid
                }
            elif isinstance(obj, dict):
                # Recursively convert dictionary values
                return {key: self._convert_neo4j_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Recursively convert list items
                return [self._convert_neo4j_types(item) for item in obj]
            elif hasattr(obj, '__dict__') and hasattr(obj, 'items'):
                # Handle other Neo4j objects that have dict-like properties
                try:
                    return {key: self._convert_neo4j_types(value) for key, value in dict(obj).items()}
                except Exception as dict_error:
                    logger.warning(f"Failed to convert dict-like object: {dict_error}")
                    # If conversion fails, convert to string as fallback
                    return str(obj)
            elif hasattr(obj, '_properties'):
                # Handle Neo4j Node/Relationship objects
                try:
                    properties = dict(obj._properties) if obj._properties else {}
                    result = {
                        "id": getattr(obj, 'element_id', getattr(obj, 'id', None)),
                        "labels": list(obj.labels) if hasattr(obj, 'labels') else [],
                        "type": obj.type if hasattr(obj, 'type') else None,
                        "properties": {key: self._convert_neo4j_types(value) for key, value in properties.items()}
                    }
                    # Remove None values
                    return {k: v for k, v in result.items() if v is not None}
                except Exception as node_error:
                    logger.warning(f"Failed to convert Neo4j node/relationship: {node_error}")
                    return str(obj)
            else:
                # Return the object as-is if it's already serializable
                # Check if it's JSON serializable
                try:
                    import json
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    return str(obj)
        except Exception as e:
            logger.error(f"Error in _convert_neo4j_types: {e}")
            return str(obj)

    async def _initialize_schema(self) -> None:
        """Initialize database schema with constraints and indexes for enhanced medical knowledge graph."""
        constraints_and_indexes = [
            # Patient constraints (Person entity)
            "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT patient_mrn_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.mrn IS UNIQUE",
            
            # Visit constraints (MedicalEncounter entity)
            "CREATE CONSTRAINT visit_id_unique IF NOT EXISTS FOR (v:Visit) REQUIRE v.id IS UNIQUE",
            
            # Condition constraints (MedicalCondition entity)
            "CREATE CONSTRAINT condition_id_unique IF NOT EXISTS FOR (c:Condition) REQUIRE c.id IS UNIQUE",
            
            # Medication constraints (Drug entity)
            "CREATE CONSTRAINT medication_id_unique IF NOT EXISTS FOR (m:Medication) REQUIRE m.id IS UNIQUE",
            
            # Test constraints (LabTest entity)
            "CREATE CONSTRAINT test_id_unique IF NOT EXISTS FOR (t:Test) REQUIRE t.id IS UNIQUE",
            
            # Procedure constraints (MedicalProcedure entity)
            "CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE",
            
            # Knowledge base constraints
            "CREATE CONSTRAINT medical_concept_id_unique IF NOT EXISTS FOR (mc:MedicalConcept) REQUIRE mc.id IS UNIQUE",
            
            # Performance indexes for common queries
            "CREATE INDEX patient_name_index IF NOT EXISTS FOR (p:Patient) ON (p.name)",
            "CREATE INDEX patient_dob_index IF NOT EXISTS FOR (p:Patient) ON (p.dob)",
            "CREATE INDEX visit_date_index IF NOT EXISTS FOR (v:Visit) ON (v.date)",
            "CREATE INDEX visit_type_index IF NOT EXISTS FOR (v:Visit) ON (v.visit_type)",
            "CREATE INDEX condition_name_index IF NOT EXISTS FOR (c:Condition) ON (c.condition_name)",
            "CREATE INDEX condition_icd_index IF NOT EXISTS FOR (c:Condition) ON (c.icd_code)",
            "CREATE INDEX condition_status_index IF NOT EXISTS FOR (c:Condition) ON (c.status)",
            "CREATE INDEX medication_name_index IF NOT EXISTS FOR (m:Medication) ON (m.medication_name)",
            "CREATE INDEX medication_rxnorm_index IF NOT EXISTS FOR (m:Medication) ON (m.rxnorm_code)",
            "CREATE INDEX test_name_index IF NOT EXISTS FOR (t:Test) ON (t.test_name)",
            "CREATE INDEX test_category_index IF NOT EXISTS FOR (t:Test) ON (t.test_category)",
            "CREATE INDEX procedure_name_index IF NOT EXISTS FOR (p:Procedure) ON (p.procedure_name)",            "CREATE INDEX procedure_cpt_index IF NOT EXISTS FOR (p:Procedure) ON (p.cpt_code)",
            
            # Entity linking indexes for knowledge base integration
            # Note: These indexes are created for all node types that might have these properties
            "CREATE INDEX entity_linking_condition_index IF NOT EXISTS FOR (c:Condition) ON (c.entity_id)",
            "CREATE INDEX entity_linking_medication_index IF NOT EXISTS FOR (m:Medication) ON (m.entity_id)",
            "CREATE INDEX concept_code_condition_index IF NOT EXISTS FOR (c:Condition) ON (c.concept_code)",
            "CREATE INDEX concept_code_medication_index IF NOT EXISTS FOR (m:Medication) ON (m.concept_code)",
            "CREATE INDEX semantic_type_condition_index IF NOT EXISTS FOR (c:Condition) ON (c.semantic_type)",
            "CREATE INDEX semantic_type_medication_index IF NOT EXISTS FOR (m:Medication) ON (m.semantic_type)",
            
            # Temporal analysis indexes
            "CREATE INDEX condition_onset_date_index IF NOT EXISTS FOR (c:Condition) ON (c.onset_date)",
            "CREATE INDEX medication_start_date_index IF NOT EXISTS FOR (m:Medication) ON (m.start_date)",
            "CREATE INDEX test_result_date_index IF NOT EXISTS FOR (t:Test) ON (t.result_date)",
            "CREATE INDEX procedure_date_index IF NOT EXISTS FOR (p:Procedure) ON (p.date)",
        ]
        
        async with self.session() as session:
            for query in constraints_and_indexes:
                try:
                    await session.run(query)
                except ClientError as e:
                    # Constraint/index might already exist
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema initialization warning: {e}")
        
        logger.info("Enhanced medical knowledge graph schema initialized successfully")
    
    async def health_check(self) -> GraphHealthCheck:
        """Perform health check on Neo4j database."""
        start_time = datetime.now()
        
        try:
            async with self.session() as session:
                result = await session.run("CALL dbms.components() YIELD name, versions, edition")
                db_info = {}
                async for record in result:
                    db_info[record["name"]] = {
                        "versions": record["versions"],
                        "edition": record["edition"]
                    }
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                return GraphHealthCheck(
                    status="healthy",
                    connection_status="connected",
                    response_time=response_time,
                    database_info=db_info
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Health check failed: {e}")
            
            return GraphHealthCheck(
                status="unhealthy",
                connection_status="disconnected",
                response_time=response_time,
                database_info={"error": str(e)}
            )
    
    async def create_patient_graph(self, request: PatientGraphRequest) -> PatientGraphResponse:
        """
        Create or update patient graph from medical data.
        
        Args:
            request: Patient graph creation request
            
        Returns:
            PatientGraphResponse with operation results        """
        start_time = datetime.now()
        nodes_created = 0
        relationships_created = 0
        
        try:
            async with self.session() as session:
                # Use execute_write for write transactions
                result = await session.execute_write(
                    self._create_patient_graph_transaction,
                    request, start_time
                )
                return result
                    
        except Exception as e:
            logger.error(f"Failed to create patient graph: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PatientGraphResponse(
                message=f"Failed to create patient graph: {str(e)}",
                success=False,
                patient_id=request.patient_id,
                nodes_created=0,
                relationships_created=0,
                graph_statistics={},
                processing_time=processing_time
            )
    
    async def _create_patient_graph_transaction(
        self, tx: AsyncTransaction, request: PatientGraphRequest, start_time: datetime
    ) -> PatientGraphResponse:
        """Execute patient graph creation within a transaction."""
        nodes_created = 0
        relationships_created = 0
        
        # Create patient node
        patient_node = await self._create_patient_node(
            tx, request.patient_id, request.visit_data
        )
        nodes_created += 1
        
        # Create visit node
        visit_node = await self._create_visit_node(
            tx, request.visit_data, request.patient_id
        )
        nodes_created += 1
        
        # Create patient-visit relationship
        await self._create_relationship(
            tx, patient_node["id"], visit_node["id"], RelationshipType.HAS_VISIT
        )
        relationships_created += 1
        
        # Create entity nodes and relationships
        for entity in request.entities:
            entity_node = await self._create_entity_node(tx, entity, visit_node["id"])
            if entity_node:
                nodes_created += 1
                
                # Create visit-entity relationship
                rel_type = self._get_entity_relationship_type(entity.get("label", "Entity"))
                await self._create_relationship(
                    tx, visit_node["id"], entity_node["id"], rel_type
                )
                relationships_created += 1
        
        # Create custom relationships
        for rel in request.relationships:
            await self._create_custom_relationship(tx, rel)
            relationships_created += 1
        
        # Get graph statistics
        stats = await self._get_patient_statistics(tx, request.patient_id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PatientGraphResponse(
            message="Patient graph created successfully",
            success=True,
            patient_id=request.patient_id,
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            graph_statistics=stats,
            processing_time=processing_time
        )
    
    async def _create_patient_node(
        self, tx: AsyncTransaction, patient_id: str, visit_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create or update patient node (Person entity)."""
        query = """
        MERGE (p:Patient:Person {id: $patient_id})
        ON CREATE SET 
            p.created_at = datetime(),
            p.name = $name,
            p.dob = $dob,
            p.gender = $gender,
            p.mrn = $mrn,
            p.address = $address,
            p.phone = $phone,
            p.email = $email,
            p.emergency_contact = $emergency_contact
        ON MATCH SET 
            p.updated_at = datetime(),
            p.name = COALESCE($name, p.name),
            p.dob = COALESCE($dob, p.dob),
            p.gender = COALESCE($gender, p.gender),
            p.mrn = COALESCE($mrn, p.mrn),
            p.address = COALESCE($address, p.address),
            p.phone = COALESCE($phone, p.phone),
            p.email = COALESCE($email, p.email),
            p.emergency_contact = COALESCE($emergency_contact, p.emergency_contact)
        RETURN p
        """
        
        result = await tx.run(query, 
            patient_id=patient_id,
            name=visit_data.get("patient_name"),
            dob=visit_data.get("patient_dob"),
            gender=visit_data.get("patient_gender"),
            mrn=visit_data.get("patient_mrn"),
            address=visit_data.get("patient_address"),
            phone=visit_data.get("patient_phone"),            email=visit_data.get("patient_email"),
            emergency_contact=visit_data.get("patient_emergency_contact")
        )
        
        record = await result.single()
        return dict(record["p"])
    
    async def _create_visit_node(
        self, tx: AsyncTransaction, visit_data: Dict[str, Any], patient_id: str
    ) -> Dict[str, Any]:
        """Create visit node (MedicalEncounter entity)."""
        visit_id = f"visit_{patient_id}_{visit_data.get('date', datetime.now().isoformat())}"
        
        query = """
        MERGE (v:Visit:MedicalEncounter {id: $visit_id})
        ON CREATE SET
            v.date = $date,
            v.visit_type = $visit_type,
            v.location = $location,
            v.provider = $provider,
            v.chief_complaint = $chief_complaint,
            v.duration_minutes = $duration_minutes,
            v.visit_status = $visit_status,
            v.notes = $notes,
            v.created_at = datetime()
        ON MATCH SET
            v.date = $date,
            v.visit_type = $visit_type,
            v.location = $location,
            v.provider = $provider,
            v.chief_complaint = $chief_complaint,
            v.duration_minutes = $duration_minutes,
            v.visit_status = $visit_status,
            v.notes = $notes,
            v.updated_at = datetime()
        RETURN v
        """
        
        result = await tx.run(query,
            visit_id=visit_id,
            date=visit_data.get("date", datetime.now().isoformat()),
            visit_type=visit_data.get("type", "unknown"),
            location=visit_data.get("location"),
            provider=visit_data.get("provider"),
            chief_complaint=visit_data.get("chief_complaint"),
            duration_minutes=visit_data.get("duration_minutes"),
            visit_status=visit_data.get("visit_status", "completed"),
            notes=visit_data.get("notes")
        )
        
        record = await result.single()
        return dict(record["v"])
    
    async def _create_entity_node(
        self, tx: AsyncTransaction, entity: Dict[str, Any], visit_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create entity node based on entity type with enhanced medical knowledge base linking."""
        entity_type = entity.get("label", "").upper()
        entity_text = entity.get("text", "")
        entity_id = f"{entity_type.lower()}_{hash(entity_text)}"
        
        # Enhanced entity type mapping to handle NER outputs
        entity_type_mapping = {
            "CONDITION": "CONDITION",
            "DIAGNOSIS": "CONDITION", 
            "DISEASE": "CONDITION",
            "MEDICAL_CONDITION": "CONDITION",
            "MEDICATION": "MEDICATION",
            "DRUG": "MEDICATION",
            "TEST": "TEST",
            "LAB_TEST": "TEST",
            "LAB_VALUE": "TEST",
            "DIAGNOSTIC_TEST": "TEST",
            "PROCEDURE": "PROCEDURE",
            "TREATMENT": "PROCEDURE",
            "MEDICAL_PROCEDURE": "PROCEDURE",
            "SURGICAL_PROCEDURE": "PROCEDURE",
            "DIAGNOSTIC_PROCEDURE": "PROCEDURE",
            "SYMPTOM": "SYMPTOM",
            "VITAL_SIGN": "TEST",  # Treat vital signs as test results
            "ANATOMICAL_STRUCTURE": "CONDITION"  # Anatomical references as part of conditions
        }
        
        mapped_type = entity_type_mapping.get(entity_type, entity_type)
        
        # Map entity types to node creation functions
        if mapped_type == "CONDITION":
            return await self._create_condition_node(tx, entity, entity_id)
        elif mapped_type == "MEDICATION":
            return await self._create_medication_node(tx, entity, entity_id)
        elif mapped_type == "TEST":
            return await self._create_test_node(tx, entity, entity_id)
        elif mapped_type == "PROCEDURE":
            return await self._create_procedure_node(tx, entity, entity_id)
        else:
            # Create generic entity node for other types
            return await self._create_generic_node(tx, entity, entity_id, mapped_type)
    
    async def _create_condition_node(
        self, tx: AsyncTransaction, entity: Dict[str, Any], entity_id: str
    ) -> Dict[str, Any]:
        """Create condition node (MedicalCondition entity)."""
        query = """
        MERGE (c:Condition:MedicalCondition {id: $entity_id})
        ON CREATE SET 
            c.condition_name = $name,
            c.text = $text,
            c.confidence = $confidence,
            c.icd_code = $icd_code,
            c.severity = $severity,
            c.status = $status,
            c.onset_date = $onset_date,
            c.primary_diagnosis = $primary_diagnosis,
            c.confidence_score = $confidence_score,
            c.normalized_text = $normalized_text,
            c.entity_id = $entity_linking_id,
            c.concept_code = $concept_code,
            c.semantic_type = $semantic_type,
            c.created_at = datetime()
        ON MATCH SET 
            c.updated_at = datetime(),
            c.confidence = CASE WHEN $confidence > c.confidence THEN $confidence ELSE c.confidence END,
            c.confidence_score = CASE WHEN $confidence_score > COALESCE(c.confidence_score, 0) 
                                 THEN $confidence_score ELSE c.confidence_score END
        RETURN c
        """
        
        result = await tx.run(query,
            entity_id=entity_id,
            name=entity.get("text", ""),
            text=entity.get("text", ""),
            confidence=entity.get("confidence", 1.0),
            icd_code=entity.get("icd_code") or entity.get("concept_code"),
            severity=entity.get("severity"),
            status=entity.get("status", "active"),
            onset_date=entity.get("onset_date"),
            primary_diagnosis=entity.get("primary_diagnosis"),
            confidence_score=entity.get("confidence", 1.0),
            normalized_text=entity.get("normalized_text"),
            entity_linking_id=entity.get("entity_id"),
            concept_code=entity.get("concept_code"),
            semantic_type=entity.get("semantic_type")
        )
        
        record = await result.single()
        return dict(record["c"])
    
    async def _create_medication_node(
        self, tx: AsyncTransaction, entity: Dict[str, Any], entity_id: str
    ) -> Dict[str, Any]:
        """Create medication node (Drug entity)."""
        query = """
        MERGE (m:Medication:Drug {id: $entity_id})
        ON CREATE SET 
            m.medication_name = $name,
            m.text = $text,
            m.confidence = $confidence,
            m.dosage = $dosage,
            m.frequency = $frequency,
            m.route = $route,
            m.rxnorm_code = $rxnorm_code,
            m.strength = $strength,
            m.form = $form,
            m.generic_name = $generic_name,
            m.brand_name = $brand_name,
            m.indication = $indication,
            m.start_date = $start_date,
            m.end_date = $end_date,
            m.normalized_text = $normalized_text,
            m.entity_id = $entity_linking_id,
            m.concept_code = $concept_code,
            m.semantic_type = $semantic_type,
            m.created_at = datetime()
        ON MATCH SET 
            m.updated_at = datetime(),
            m.confidence = CASE WHEN $confidence > m.confidence THEN $confidence ELSE m.confidence END
        RETURN m
        """
        
        result = await tx.run(query,
            entity_id=entity_id,
            name=entity.get("text", ""),
            text=entity.get("text", ""),
            confidence=entity.get("confidence", 1.0),
            dosage=entity.get("dosage"),
            frequency=entity.get("frequency"),
            route=entity.get("route"),
            rxnorm_code=entity.get("rxnorm_code") or entity.get("concept_code"),
            strength=entity.get("strength"),
            form=entity.get("form"),
            generic_name=entity.get("generic_name"),
            brand_name=entity.get("brand_name"),
            indication=entity.get("indication"),
            start_date=entity.get("start_date"),
            end_date=entity.get("end_date"),
            normalized_text=entity.get("normalized_text"),
            entity_linking_id=entity.get("entity_id"),
            concept_code=entity.get("concept_code"),
            semantic_type=entity.get("semantic_type")
        )
        
        record = await result.single()
        return dict(record["m"])
    
    async def _create_test_node(
        self, tx: AsyncTransaction, entity: Dict[str, Any], entity_id: str
    ) -> Dict[str, Any]:
        """Create test node (LabTest entity)."""
        query = """
        MERGE (t:Test:LabTest {id: $entity_id})
        ON CREATE SET 
            t.test_name = $name,
            t.text = $text,
            t.confidence = $confidence,
            t.value = $value,
            t.unit = $unit,
            t.reference_range = $reference_range,
            t.status = $status,
            t.ordered_date = $ordered_date,
            t.result_date = $result_date,
            t.test_category = $test_category,
            t.abnormal_flag = $abnormal_flag,
            t.interpretation = $interpretation,
            t.methodology = $methodology,
            t.ordering_provider = $ordering_provider,
            t.normalized_text = $normalized_text,
            t.entity_id = $entity_linking_id,
            t.concept_code = $concept_code,
            t.semantic_type = $semantic_type,
            t.created_at = datetime()
        ON MATCH SET 
            t.updated_at = datetime(),
            t.confidence = CASE WHEN $confidence > t.confidence THEN $confidence ELSE t.confidence END
        RETURN t
        """
        
        result = await tx.run(query,
            entity_id=entity_id,
            name=entity.get("text", ""),
            text=entity.get("text", ""),
            confidence=entity.get("confidence", 1.0),
            value=entity.get("value"),
            unit=entity.get("unit"),
            reference_range=entity.get("reference_range"),
            status=entity.get("status", "completed"),
            ordered_date=entity.get("ordered_date"),
            result_date=entity.get("result_date"),
            test_category=entity.get("test_category", "lab"),
            abnormal_flag=entity.get("abnormal_flag"),
            interpretation=entity.get("interpretation"),
            methodology=entity.get("methodology"),
            ordering_provider=entity.get("ordering_provider"),
            normalized_text=entity.get("normalized_text"),
            entity_linking_id=entity.get("entity_id"),
            concept_code=entity.get("concept_code"),
            semantic_type=entity.get("semantic_type")
        )
        
        record = await result.single()
        return dict(record["t"])
    
    async def _create_procedure_node(
        self, tx: AsyncTransaction, entity: Dict[str, Any], entity_id: str
    ) -> Dict[str, Any]:
        """Create procedure node (MedicalProcedure entity)."""
        query = """
        MERGE (p:Procedure:MedicalProcedure {id: $entity_id})
        ON CREATE SET 
            p.procedure_name = $name,
            p.text = $text,
            p.confidence = $confidence,
            p.cpt_code = $cpt_code,
            p.description = $description,
            p.date = $date,
            p.status = $status,
            p.duration_minutes = $duration_minutes,
            p.performing_provider = $performing_provider,
            p.location = $location,
            p.indication = $indication,
            p.normalized_text = $normalized_text,
            p.entity_id = $entity_linking_id,
            p.concept_code = $concept_code,
            p.semantic_type = $semantic_type,
            p.created_at = datetime()
        ON MATCH SET 
            p.updated_at = datetime(),
            p.confidence = CASE WHEN $confidence > p.confidence THEN $confidence ELSE p.confidence END
        RETURN p
        """
        
        result = await tx.run(query,
            entity_id=entity_id,
            name=entity.get("text", ""),
            text=entity.get("text", ""),
            confidence=entity.get("confidence", 1.0),
            cpt_code=entity.get("cpt_code") or entity.get("concept_code"),
            description=entity.get("description"),
            date=entity.get("date"),
            status=entity.get("status", "completed"),
            duration_minutes=entity.get("duration_minutes"),
            performing_provider=entity.get("performing_provider"),
            location=entity.get("location"),
            indication=entity.get("indication"),
            normalized_text=entity.get("normalized_text"),
            entity_linking_id=entity.get("entity_id"),
            concept_code=entity.get("concept_code"),
            semantic_type=entity.get("semantic_type")
        )
        
        record = await result.single()
        return dict(record["p"])
    
    async def _create_generic_node(
        self, tx: AsyncTransaction, entity: Dict[str, Any], entity_id: str, node_type: str
    ) -> Dict[str, Any]:
        """Create generic entity node."""
        query = f"""
        MERGE (n:{node_type} {{id: $entity_id}})
        ON CREATE SET 
            n.name = $name,
            n.text = $text,
            n.confidence = $confidence,
            n.label = $label,
            n.created_at = datetime()
        ON MATCH SET 
            n.updated_at = datetime(),
            n.confidence = CASE WHEN $confidence > n.confidence THEN $confidence ELSE n.confidence END
        RETURN n
        """
        
        result = await tx.run(query,
            entity_id=entity_id,
            name=entity.get("text", ""),
            text=entity.get("text", ""),
            confidence=entity.get("confidence", 1.0),
            label=entity.get("label", "")
        )
        
        record = await result.single()
        return dict(record["n"])
    
    def _get_entity_relationship_type(self, entity_label: str) -> RelationshipType:
        """Map entity label to appropriate relationship type."""
        entity_mapping = {
            "CONDITION": RelationshipType.DIAGNOSED_WITH,
            "DIAGNOSIS": RelationshipType.DIAGNOSED_WITH,
            "DISEASE": RelationshipType.DIAGNOSED_WITH,
            "MEDICATION": RelationshipType.PRESCRIBED,
            "DRUG": RelationshipType.PRESCRIBED,
            "TEST": RelationshipType.PERFORMED,
            "LAB_TEST": RelationshipType.PERFORMED,
            "DIAGNOSTIC_TEST": RelationshipType.PERFORMED,
            "PROCEDURE": RelationshipType.UNDERWENT,
            "TREATMENT": RelationshipType.UNDERWENT,
        }
        
        return entity_mapping.get(entity_label.upper(), RelationshipType.RELATED_TO)
    
    async def _create_relationship(
        self, tx: AsyncTransaction, source_id: str, target_id: str, 
        rel_type: RelationshipType, properties: Dict[str, Any] = None
    ) -> None:
        """Create relationship between nodes."""
        if properties is None:
            properties = {}
        
        query = f"""
        MATCH (a), (b)
        WHERE a.id = $source_id AND b.id = $target_id
        MERGE (a)-[r:{rel_type.value}]->(b)
        ON CREATE SET 
            r.created_at = datetime(),
            r.confidence = $confidence
        """
        
        # Add properties to the query
        if properties:
            property_sets = [f"r.{key} = ${key}" for key in properties.keys()]
            query += ", " + ", ".join(property_sets)
        
        await tx.run(query, 
            source_id=source_id, 
            target_id=target_id,
            confidence=properties.get("confidence", 1.0),
            **properties
        )
    
    async def _create_custom_relationship(
        self, tx: AsyncTransaction, relationship: Dict[str, Any]
    ) -> None:
        """Create custom relationship from relationship data."""
        await self._create_relationship(
            tx,            relationship.get("source_id"),
            relationship.get("target_id"),
            RelationshipType(relationship.get("type", "RELATED_TO")),
            relationship.get("properties", {})
        )
    
    async def _get_patient_statistics(
        self, tx: AsyncTransaction, patient_id: str
    ) -> Dict[str, Any]:
        """Get statistics for a patient's graph."""
        query = """
        MATCH (p:Patient {id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_VISIT]->(v:Visit)
        OPTIONAL MATCH (v)-[:DIAGNOSED_WITH]->(c:Condition)
        OPTIONAL MATCH (v)-[:PRESCRIBED]->(m:Medication)
        OPTIONAL MATCH (v)-[:PERFORMED]->(t:Test)
        OPTIONAL MATCH (v)-[:UNDERWENT]->(pr:Procedure)
        RETURN 
            count(DISTINCT v) as visits,
            count(DISTINCT c) as conditions,
            count(DISTINCT m) as medications,
            count(DISTINCT t) as tests,
            count(DISTINCT pr) as procedures
        """
        
        result = await tx.run(query, patient_id=patient_id)
        record = await result.single()
        return {
            "total_visits": record["visits"],
            "conditions": record["conditions"],  
            "medications": record["medications"],
            "lab_tests": record["tests"],
            "procedures": record["procedures"]
        }
    
    async def execute_query(self, request: GraphQueryRequest) -> GraphQueryResponse:
        """Execute Cypher query and return results with proper async result handling."""
        start_time = datetime.now()
        
        try:
            # Validate query for potential slice syntax issues
            if '..' in request.query and '[..' in request.query:
                logger.warning("Detected potential invalid slice syntax in Cypher query")
                # Fix common slice syntax issues
                fixed_query = request.query.replace('[..', '[0..')
                logger.info(f"Auto-corrected slice syntax in query")
                request.query = fixed_query
            
            async with self.session() as session:
                logger.debug(f"Executing query: {request.query[:100]}...")
                logger.debug(f"With parameters: {request.parameters}")
                
                # Execute the query
                result = await session.run(request.query, request.parameters or {})
                
                records = []
                columns = []                # Use a try-finally to ensure proper cleanup
                try:
                    # Collect all records synchronously to avoid async generator issues
                    collected_records = []
                      # Add comprehensive error handling for slice-related issues
                    try:
                        # Use a different approach: collect records in a list first
                        async for record in result:
                            try:
                                # Safely convert record to dict with comprehensive error handling
                                record_dict = {}
                                
                                # First attempt: safe conversion with type checking
                                for key in record.keys():
                                    try:
                                        value = record[key]
                                        # Handle slice objects that might come from invalid Cypher syntax
                                        if hasattr(value, '__class__') and 'slice' in str(type(value)).lower():
                                            logger.warning(f"Found slice object in Neo4j result for key '{key}'. This indicates invalid Cypher slice syntax.")
                                            record_dict[key] = str(value)  # Convert to string representation
                                        elif isinstance(value, slice):
                                            logger.warning(f"Found slice object for key '{key}': {value}")
                                            record_dict[key] = f"slice({value.start}, {value.stop}, {value.step})"
                                        else:
                                            record_dict[key] = value
                                    except Exception as field_error:
                                        logger.warning(f"Error processing field '{key}': {field_error}")
                                        try:
                                            # Fallback: convert to string
                                            record_dict[key] = str(record[key])
                                        except Exception:
                                            record_dict[key] = f"<error_processing_field_{key}>"
                                
                                collected_records.append(record_dict)
                                
                            except Exception as record_error:
                                logger.error(f"Error converting Neo4j record to dict: {record_error}")
                                # Try most basic approach - convert entire record to dict with fallbacks
                                try:
                                    # Attempt basic dict conversion with extensive error handling
                                    basic_record = {}
                                    try:
                                        # Try to get record data without using dict() constructor
                                        for i, key in enumerate(record.keys()):
                                            try:
                                                basic_record[key] = record.value(i)
                                            except Exception:
                                                try:
                                                    basic_record[key] = record[i]
                                                except Exception:
                                                    basic_record[key] = f"<unable_to_extract_value_{i}>"
                                    except Exception:
                                        # Last resort: create minimal record
                                        basic_record = {"error": f"Record processing failed: {record_error}"}
                                    
                                    collected_records.append(basic_record)
                                except Exception:
                                    logger.error(f"Failed to process record completely, skipping")
                                    continue
                            
                            if request.limit and len(collected_records) >= request.limit:
                                break
                    
                    except Exception as iteration_error:
                        logger.error(f"Error during result iteration: {iteration_error}")
                        # Even if iteration fails, continue with what we have
                    
                    # Process the collected records
                    for record_data in collected_records:
                        if not columns:
                            columns = list(record_data.keys())
                        
                        record_dict = {}
                        for key, value in record_data.items():
                            try:
                                record_dict[key] = self._convert_neo4j_types(value)
                            except Exception as convert_error:
                                logger.warning(f"Failed to convert field {key}: {convert_error}")
                                record_dict[key] = str(value)
                        
                        records.append(record_dict)
                    
                    logger.debug(f"Successfully processed {len(records)} records")
                    
                except Exception as processing_error:
                    logger.error(f"Error during result processing: {processing_error}", exc_info=True)
                    
                    # Even if processing fails, try to return what we have
                    if not records:
                        records = []
                        columns = []
                
                finally:
                    # Ensure result is properly closed/consumed
                    try:
                        # Try to consume remaining results to clean up
                        await result.consume()
                    except Exception as cleanup_error:
                        logger.debug(f"Result cleanup error (non-critical): {cleanup_error}")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return GraphQueryResponse(
                    message="Query executed successfully",
                    success=True,
                    data={"query": request.query, "parameters": request.parameters or {}},
                    metadata={
                        "query_type": "cypher",
                        "limit_applied": request.limit,
                        "columns_found": len(columns)
                    },
                    processing_time=processing_time,
                    timestamp=datetime.now(),
                    results=records,
                    columns=columns,
                    result_count=len(records)
                )
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query execution failed: {e}", exc_info=True)
            
            return GraphQueryResponse(
                message=f"Query execution failed: {str(e)}",
                success=False,
                data={"query": request.query, "parameters": request.parameters or {}},
                metadata={
                    "error_type": type(e).__name__,
                    "query_type": "cypher"
                },
                processing_time=processing_time,
                timestamp=datetime.now(),
                results=[],
                columns=[],
                result_count=0
            )
    
    async def get_patient_graph(self, patient_id: str) -> Dict[str, Any]:
        """Retrieve complete patient graph."""
        query = """
        MATCH (p:Patient {id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_VISIT]->(v:Visit)
        OPTIONAL MATCH (v)-[r]->(entity)
        RETURN p, collect(DISTINCT v) as visits, 
               collect(DISTINCT {entity: entity, relationship: type(r)}) as entities
        """
        
        try:
            async with self.session() as session:
                result = await session.run(query, patient_id=patient_id)
                record = await result.single()
                
                if not record:
                    return {"error": "Patient not found"}
                
                # Convert Neo4j objects to serializable Python types
                patient = self._convert_neo4j_types(dict(record["p"])) if record["p"] else {}
                visits = self._convert_neo4j_types([dict(v) for v in record["visits"] if v]) if record["visits"] else []
                entities = self._convert_neo4j_types(record["entities"]) if record["entities"] else []
                
                # Get patient statistics in the same session
                stats_query = """
                MATCH (p:Patient {id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_VISIT]->(v:Visit)
                OPTIONAL MATCH (p)-[:DIAGNOSED_WITH]->(c:Condition)
                OPTIONAL MATCH (p)-[:PRESCRIBED]->(m:Medication)
                OPTIONAL MATCH (v)-[:PERFORMED]->(t:Test)
                OPTIONAL MATCH (v)-[:UNDERWENT]->(pr:Procedure)
                RETURN 
                    count(DISTINCT v) as visits,
                    count(DISTINCT c) as conditions,
                    count(DISTINCT m) as medications,
                    count(DISTINCT t) as tests,
                    count(DISTINCT pr) as procedures
                """
                
                stats_result = await session.run(stats_query, patient_id=patient_id)
                stats_record = await stats_result.single()
                
                summary = {
                    "total_visits": stats_record["visits"] if stats_record else 0,
                    "conditions": stats_record["conditions"] if stats_record else 0,
                    "medications": stats_record["medications"] if stats_record else 0,
                    "lab_tests": stats_record["tests"] if stats_record else 0,
                    "procedures": stats_record["procedures"] if stats_record else 0
                }
                
                return {
                    "patient": patient,
                    "visits": visits,
                    "entities": entities,
                    "summary": summary
                }
                
        except Exception as e:
            logger.error(f"Error retrieving patient graph for {patient_id}: {e}")
            return {"error": f"Failed to retrieve patient graph: {str(e)}"}
    
    async def get_graph_statistics(self) -> GraphStatistics:
        """Get overall graph statistics."""
        query = """
        MATCH (n)
        OPTIONAL MATCH ()-[r]->()
        RETURN 
            count(DISTINCT n) as total_nodes,
            count(DISTINCT r) as total_relationships,
            collect(DISTINCT labels(n)) as node_labels,
            collect(DISTINCT type(r)) as relationship_types
        """
        
        async with self.session() as session:
            result = await session.run(query)
            record = await result.single()
            
            # Count nodes by type
            node_type_query = """
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            """
            
            node_result = await session.run(node_type_query)
            node_types = {}
            async for node_record in node_result:
                node_types[node_record["node_type"]] = node_record["count"]
            
            # Count relationships by type
            rel_type_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            """
            
            rel_result = await session.run(rel_type_query)
            relationship_types = {}
            async for rel_record in rel_result:
                relationship_types[rel_record["rel_type"]] = rel_record["count"]
            
            return GraphStatistics(
                total_nodes=record["total_nodes"],
                total_relationships=record["total_relationships"],
                node_types=node_types,                relationship_types=relationship_types,
                patients_count=node_types.get("Patient", 0),
                visits_count=node_types.get("Visit", 0)
            )
    
    async def create_temporal_relationship(
        self, patient_id: str, entity1_id: str, entity2_id: str, 
        relationship_type: str, temporal_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create temporal relationships between medical entities."""
        try:
            query = """
            MATCH (e1 {id: $entity1_id}), (e2 {id: $entity2_id})
            WHERE EXISTS((p:Patient {id: $patient_id})-[:HAS_VISIT]->()-[:DIAGNOSED_WITH|:PRESCRIBED|:PERFORMED|:UNDERWENT]->(e1))
              AND EXISTS((p:Patient {id: $patient_id})-[:HAS_VISIT]->()-[:DIAGNOSED_WITH|:PRESCRIBED|:PERFORMED|:UNDERWENT]->(e2))
            CREATE (e1)-[r:TEMPORAL_RELATIONSHIP {
                type: $relationship_type,
                temporal_order: $temporal_order,
                time_difference: $time_difference,
                causality_confidence: $causality_confidence,
                created_at: datetime()
            }]->(e2)
            RETURN r.type as relationship_type, r.temporal_order as temporal_order, 
                   r.time_difference as time_difference, r.causality_confidence as causality_confidence
            """
            
            async with self.session() as session:
                result = await session.run(query,
                    patient_id=patient_id,
                    entity1_id=entity1_id,
                    entity2_id=entity2_id,
                    relationship_type=relationship_type,
                    temporal_order=temporal_context.get("temporal_order"),
                    time_difference=temporal_context.get("time_difference"),
                    causality_confidence=temporal_context.get("causality_confidence", 0.5)
                )
                
                # Use result.data() instead of result.single() to avoid generator issues
                records = await result.data()
                
                if records and len(records) > 0:
                    record_data = records[0]
                    return {
                        "relationship_type": record_data.get("relationship_type"),
                        "temporal_order": record_data.get("temporal_order"),
                        "time_difference": record_data.get("time_difference"),
                        "causality_confidence": record_data.get("causality_confidence"),
                        "entity1_id": entity1_id,
                        "entity2_id": entity2_id,
                        "patient_id": patient_id
                    }
                else:
                    logger.warning(f"No temporal relationship created between {entity1_id} and {entity2_id} for patient {patient_id}")
                    return {
                        "error": "No relationship created - entities may not be connected to the patient",
                        "entity1_id": entity1_id,
                        "entity2_id": entity2_id,
                        "patient_id": patient_id
                    }
                    
        except Exception as e:
            logger.error(f"Error creating temporal relationship: {e}")
            return {
                "error": str(e),                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "patient_id": patient_id
            }

    async def analyze_patient_timeline(self, patient_id: str) -> Dict[str, Any]:
        """Analyze patient's medical timeline for patterns and insights."""
        query = """
        MATCH (p:Patient {id: $patient_id})-[:HAS_VISIT]->(v:Visit)
        OPTIONAL MATCH (v)-[r1:DIAGNOSED_WITH]->(c:Condition)
        OPTIONAL MATCH (v)-[r2:PRESCRIBED]->(m:Medication)
        OPTIONAL MATCH (v)-[r3:PERFORMED]->(t:Test)
        OPTIONAL MATCH (v)-[r4:UNDERWENT]->(pr:Procedure)
        RETURN v.date as visit_date, v.visit_type as visit_type,
               collect(DISTINCT c.condition_name) as conditions,
               collect(DISTINCT m.medication_name) as medications,
               collect(DISTINCT t.test_name) as tests,
               collect(DISTINCT pr.procedure_name) as procedures
        ORDER BY v.date ASC
        """
        
        async with self.session() as session:
            result = await session.run(query, patient_id=patient_id)
            timeline = []
            
            async for record in result:
                timeline.append({
                    "visit_date": record["visit_date"],
                    "visit_type": record["visit_type"],
                    "conditions": record["conditions"],
                    "medications": record["medications"],
                    "tests": record["tests"],
                    "procedures": record["procedures"]
                })
            
            return {
                "patient_id": patient_id,
                "timeline": timeline,
                "total_visits": len(timeline),
                "analysis_timestamp": datetime.now().isoformat()
            }

    async def find_cross_patient_patterns(
        self, condition_name: str, limit: int = 100
    ) -> Dict[str, Any]:
        """Find patterns across patients with similar conditions."""
        query = """
        MATCH (c:Condition {condition_name: $condition_name})<-[:DIAGNOSED_WITH]-(v:Visit)<-[:HAS_VISIT]-(p:Patient)
        OPTIONAL MATCH (v)-[:PRESCRIBED]->(m:Medication)
        OPTIONAL MATCH (v)-[:PERFORMED]->(t:Test)
        RETURN p.id as patient_id, 
               v.date as diagnosis_date,
               collect(DISTINCT m.medication_name) as common_medications,
               collect(DISTINCT t.test_name) as common_tests
        ORDER BY v.date DESC
        LIMIT $limit
        """
        
        async with self.session() as session:
            result = await session.run(query, 
                condition_name=condition_name, 
                limit=limit
            )
            
            patterns = []
            all_medications = []
            all_tests = []
            
            async for record in result:
                patterns.append({
                    "patient_id": record["patient_id"],
                    "diagnosis_date": record["diagnosis_date"],
                    "medications": record["common_medications"],
                    "tests": record["common_tests"]
                })
                all_medications.extend(record["common_medications"])
                all_tests.extend(record["common_tests"])
              # Analyze common patterns
            from collections import Counter
            medication_frequency = Counter(all_medications)
            test_frequency = Counter(all_tests)
            
            return {
                "condition": condition_name,
                "patient_count": len(patterns),
                "patterns": patterns,
                "common_medications": dict(medication_frequency.most_common(10)),
                "common_tests": dict(test_frequency.most_common(10)),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def expand_knowledge_base(
        self, medical_concepts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Expand knowledge base with new medical concepts and relationships."""
        nodes_created = 0
        relationships_created = 0
        errors = []
        
        async with self.session() as session:
            try:
                for concept in medical_concepts:
                    try:
                        # Validate required fields
                        if not concept.get("id") or not concept.get("name"):
                            errors.append(f"Concept missing required fields: {concept}")
                            continue
                        
                        # Create concept node using session.run (not transaction)
                        concept_query = """
                        MERGE (c:MedicalConcept {id: $concept_id})
                        ON CREATE SET 
                            c.name = $name,
                            c.category = $category,
                            c.description = $description,
                            c.source = $source,
                            c.concept_code = $concept_code,
                            c.confidence = $confidence,
                            c.created_at = datetime()
                        ON MATCH SET
                            c.updated_at = datetime(),
                            c.confidence = CASE WHEN $confidence > c.confidence 
                                           THEN $confidence ELSE c.confidence END
                        RETURN c.id as created_id
                        """
                        
                        result = await session.run(concept_query,
                            concept_id=concept.get("id"),
                            name=concept.get("name"),
                            category=concept.get("category", "general"),
                            description=concept.get("description", ""),
                            source=concept.get("source", "unknown"),
                            concept_code=concept.get("concept_code"),
                            confidence=concept.get("confidence", 1.0)
                        )
                        
                        # Properly consume the result
                        records = await result.data()
                        if records:
                            nodes_created += 1
                            logger.info(f"Created/updated concept: {concept.get('name')}")
                        
                        # Create relationships to existing entities if specified
                        if concept.get("related_entities"):
                            for related_entity in concept["related_entities"]:
                                try:
                                    rel_query = """
                                    MATCH (c:MedicalConcept {id: $concept_id})
                                    MATCH (e {id: $entity_id})
                                    MERGE (c)-[r:KNOWLEDGE_RELATES_TO]->(e)
                                    ON CREATE SET 
                                        r.relationship_type = $rel_type,
                                        r.confidence = $confidence,
                                        r.created_at = datetime()
                                    RETURN r.relationship_type as rel_type
                                    """
                                    
                                    rel_result = await session.run(rel_query,
                                        concept_id=concept.get("id"),
                                        entity_id=related_entity.get("entity_id"),
                                        rel_type=related_entity.get("relationship_type", "RELATED_TO"),
                                        confidence=related_entity.get("confidence", 0.8)
                                    )
                                    
                                    # Properly consume the relationship result
                                    rel_records = await rel_result.data()
                                    if rel_records:
                                        relationships_created += 1
                                        logger.info(f"Created relationship: {concept.get('id')} -> {related_entity.get('entity_id')}")
                                
                                except Exception as rel_error:
                                    logger.error(f"Failed to create relationship for concept {concept.get('id')}: {rel_error}")
                                    errors.append(f"Relationship error for {concept.get('id')}: {str(rel_error)}")
                    
                    except Exception as concept_error:
                        logger.error(f"Failed to process concept {concept.get('id', 'unknown')}: {concept_error}")
                        errors.append(f"Concept error: {str(concept_error)}")
                
                logger.info(f"Knowledge base expansion completed: {nodes_created} nodes, {relationships_created} relationships")
                
            except Exception as e:
                logger.error(f"Knowledge base expansion failed: {e}")
                raise Exception(f"Failed to expand knowledge base: {str(e)}")
        
        return {
            "message": "Knowledge base expanded successfully",
            "nodes_created": nodes_created,
            "relationships_created": relationships_created,
            "concepts_processed": len(medical_concepts),
            "errors": errors,
            "success": len(errors) == 0
        }

    async def get_patient_insights(self, patient_id: str) -> Dict[str, Any]:
        """Generate comprehensive insights for a patient's medical graph."""
        # Get basic patient statistics
        stats = await self.get_patient_graph(patient_id)
        
        # Get timeline analysis
        timeline = await self.analyze_patient_timeline(patient_id)
        
        # Find potential drug interactions
        drug_interactions_query = """
        MATCH (p:Patient {id: $patient_id})-[:HAS_VISIT]->(v:Visit)-[:PRESCRIBED]->(m1:Medication)
        MATCH (v)-[:PRESCRIBED]->(m2:Medication)
        WHERE m1.id <> m2.id
        OPTIONAL MATCH (m1)-[:INTERACTS_WITH]-(m2)
        RETURN m1.medication_name as drug1, m2.medication_name as drug2,
               EXISTS((m1)-[:INTERACTS_WITH]-(m2)) as has_interaction
        """
        
        # Find care gaps
        care_gaps_query = """
        MATCH (p:Patient {id: $patient_id})-[:HAS_VISIT]->(v:Visit)-[:DIAGNOSED_WITH]->(c:Condition)
        WHERE c.status = 'active' 
        OPTIONAL MATCH (v)-[:PRESCRIBED]->(m:Medication)
        WHERE (c)-[:TREATED_WITH]->(m)
        RETURN c.condition_name as condition, 
               count(m) as prescribed_treatments,
               CASE WHEN count(m) = 0 THEN 'Treatment Gap' ELSE 'Treated' END as status
        """
        
        async with self.session() as session:
            # Get drug interactions
            drug_result = await session.run(drug_interactions_query, patient_id=patient_id)
            interactions = []
            async for record in drug_result:
                interactions.append({
                    "drug1": record["drug1"],
                    "drug2": record["drug2"],
                    "has_interaction": record["has_interaction"]
                })
            
            # Get care gaps
            gaps_result = await session.run(care_gaps_query, patient_id=patient_id)
            care_gaps = []
            async for record in gaps_result:
                care_gaps.append({
                    "condition": record["condition"],
                    "prescribed_treatments": record["prescribed_treatments"],
                    "status": record["status"]
                })
        
        return {
            "patient_id": patient_id,
            "graph_statistics": stats.get("summary", {}),
            "timeline_analysis": timeline,
            "potential_drug_interactions": interactions,
            "care_gaps": care_gaps,
            "insights_generated_at": datetime.now().isoformat()
        }
    

# Global client instance for singleton pattern
_client_instance = None


def get_graph_client_from_config() -> Neo4jGraphClient:
    """
    Factory function to create a Neo4j graph client from configuration.
    Returns a singleton instance to avoid multiple connections.
    
    Returns:
        Neo4jGraphClient: Configured Neo4j client instance using environment configuration
    """
    global _client_instance
    
    # Return existing instance if available
    if _client_instance is not None:
        return _client_instance
        
    # Create new instance and store as singleton
    _client_instance = Neo4jGraphClient()
    return _client_instance
