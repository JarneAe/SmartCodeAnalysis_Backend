from pydantic import BaseModel, Field


class CodeRequest(BaseModel):
    code_snippet: str = Field(
        default="""
        public Long getWarehouseIdByLicensePlate(String licensePlate) {\n    try {\n        Pair<Long, String> data = appointmentRetrievalService.getClientIdAndMineralByLicensePlate(licensePlate);\n        if (data == null) {\n            throw new NoSuchElementException(\"No data found for license plate: \" + licensePlate);\n        }\n\n        Long sellerId = data.getFirst();\n        String mineralName = data.getSecond();\n\n        log.info(\"Fetching Warehouse id for seller with id {} and mineral {}\", sellerId, mineralName);\n\n        String url = String.format(warehouseApiUrl, sellerId, mineralName);\n\n        ResponseEntity<Long> response = restTemplate.getForEntity(url, Long.class);\n\n        if (response.getStatusCode().is2xxSuccessful()) {\n            Long warehouseId = response.getBody();\n            log.info(\"Warehouse id retrieved: {}\", warehouseId);\n            return warehouseId;\n        } else {\n            log.error(\"Failed to retrieve warehouse id. Status code: {}\", response.getStatusCode());\n            throw new CouldNotRetrieveWarehouseException(\"Failed to retrieve warehouse id. Status code: \" + response.getStatusCode());\n        }\n    } catch (NoSuchElementException e) {\n        log.error(\"Error retrieving warehouse id for license plate: {}\", licensePlate, e);\n        throw new CouldNotRetrieveWarehouseException(\"Error retrieving warehouse id for license plate: \" + licensePlate, e);\n    }\n}
        """
    )
    user_role: str = Field(default="Sales Manager")