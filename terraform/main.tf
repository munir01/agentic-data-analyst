terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  region  = "us-east-1"
  profile = "agentic-ai-dev"
}

data "aws_caller_identity" "current" {}

data "aws_ecr_repository" "olist_agent" {
  name = "olist-agent"
}

data "aws_s3_bucket" "olist_data" {
  bucket = "munir-agentic-data-analyst-638869575937"
}

data "aws_iam_role" "github_actions" {
  name = "GitHubActionsOlistAgent"
}

output "aws_account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "ecr_repository_url" {
  value = data.aws_ecr_repository.olist_agent.repository_url
}

output "s3_bucket_name" {
  value = data.aws_s3_bucket.olist_data.bucket
}

output "github_actions_role_arn" {
  value = data.aws_iam_role.github_actions.arn
}

resource "aws_ecr_lifecycle_policy" "olist_agent" {
  repository = data.aws_ecr_repository.olist_agent.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only the 5 newest images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

data "aws_iam_role" "ecs_task" {
  name = "olist-agent-task-role"
}

resource "aws_ecs_express_gateway_service" "olist_agent" {
  service_name = "olist-agent"

  execution_role_arn      = "arn:aws:iam::638869575937:role/service-role/ecsTaskExecutionRole"
  infrastructure_role_arn = "arn:aws:iam::638869575937:role/service-role/ecsInfrastructureRoleForExpressServices"
  task_role_arn           = data.aws_iam_role.ecs_task.arn

  cpu    = "1024"
  memory = "2048"

  health_check_path = "/health"

  primary_container {
    image          = "${data.aws_ecr_repository.olist_agent.repository_url}:latest"
    container_port = 8000
    secret {
      name       = "API_KEY"
      value_from = "arn:aws:secretsmanager:us-east-1:638869575937:secret:olist-agent-api-key-VAvv1J"
    }
  }

  scaling_target {
    min_task_count            = 1
    max_task_count            = 1
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = 60
  }

}