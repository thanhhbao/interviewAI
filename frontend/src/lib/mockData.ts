import {
  CandidateProfile,
  TargetJob,
  ResumeSuggestion,
  InterviewQuestion,
  InterviewReport,
} from './types/interview';

export const initialCandidateProfile: CandidateProfile = {
  name: 'Alex Nguyen',
  title: 'Senior Software Engineer',
  resumeFileName: 'Alex_Nguyen_SWE_Resume.pdf',
  lastUpdated: 'Updated 3 days ago',
  summary:
    'Experienced Backend & Distributed Systems Engineer with 5+ years building high-throughput microservices, REST/gRPC APIs, and event-driven data pipelines. Proficient in Python, FastAPI, PostgreSQL, Docker, and AWS cloud infrastructure.',
  experience: [
    {
      id: 'exp-1',
      company: 'CloudScale Systems',
      role: 'Senior Backend Engineer',
      duration: '2022 - Present · 2 yrs 4 mos',
      highlights: [
        'Architected real-time ingestion service processing 40M+ events/day using FastAPI, Kafka, and TimescaleDB, lowering p99 latency from 180ms to 42ms.',
        'Migrated monolithic data pipeline to containerized Kubernetes microservices on AWS EKS, reducing compute overhead by 28%.',
        'Implemented Redis distributed caching layer and database query indexing, cutting PostgreSQL CPU load by 35% during peak traffic spikes.',
      ],
    },
    {
      id: 'exp-2',
      company: 'TechFlow Labs',
      role: 'Software Engineer',
      duration: '2020 - 2022 · 2 yrs',
      highlights: [
        'Engineered customer-facing RESTful APIs in Python/Django and PostgreSQL supporting 250k daily active users.',
        'Designed asynchronous worker queues with Celery and Redis to handle batch email notifications and external webhook deliveries.',
        'Built automated CI/CD deployment pipelines using GitHub Actions, reducing release cycle time from 3 days to under 20 minutes.',
      ],
    },
  ],
  education: [
    {
      id: 'edu-1',
      school: 'University of California, Irvine',
      degree: 'B.S. in Computer Science',
      year: '2016 - 2020',
    },
  ],
  skills: [
    'Python',
    'FastAPI',
    'PostgreSQL',
    'Docker',
    'AWS (ECS, S3, RDS)',
    'Redis',
    'REST APIs',
    'Microservices',
    'Git',
    'CI/CD',
    'Linux',
  ],
  projects: [
    {
      id: 'proj-1',
      name: 'OmniCache Distributed KV Store',
      description:
        'A lightweight distributed in-memory key-value cache written in Python with Raft consensus and LRU eviction policy.',
      tech: ['Python', 'AsyncIO', 'Sockets', 'Raft Algorithm'],
    },
    {
      id: 'proj-2',
      name: 'RateGate API Limiter',
      description:
        'High-performance token-bucket rate limiter middleware for FastAPI backed by Redis Lua scripts.',
      tech: ['FastAPI', 'Redis', 'Python', 'Docker'],
    },
  ],
};

export const initialTargetJob: TargetJob = {
  title: 'Software Engineer - Platform & Core APIs',
  company: 'Nova Technologies',
  location: 'San Francisco, CA (Hybrid / Remote Option)',
  expectedDurationMinutes: 25,
  questionCount: 4,
  description:
    'Nova Technologies is building next-generation cloud infrastructure orchestration software. We are looking for a Software Engineer to join our Core Platform team to design, implement, and scale distributed REST and gRPC services powering thousands of mission-critical developer workflows.',
  requiredSkills: [
    'Python',
    'FastAPI or similar asynchronous web framework',
    'PostgreSQL & relational database tuning',
    'Docker & containerized workloads',
    'AWS cloud services (ECS/EKS, RDS, SQS)',
    'Distributed systems design & REST API standards',
  ],
  preferredSkills: [
    'Kubernetes orchestration & Helm',
    'Event streaming with Kafka or RabbitMQ',
    'Redis caching patterns & lock management',
    'Observability with OpenTelemetry / Datadog',
  ],
  matchScore: 88,
  matchedSkills: [
    'Python',
    'FastAPI',
    'PostgreSQL',
    'Docker',
    'AWS Cloud',
    'REST APIs',
    'Microservices',
  ],
  partiallyMatchedSkills: ['Redis Caching', 'Kubernetes'],
  missingSkills: ['Kafka Event Streaming', 'OpenTelemetry / Datadog'],
  focusAreas: [
    'Distributed caching & lock contention',
    'PostgreSQL connection pooling under 50k RPS',
    'Algorithmic problem solving and data structures',
    'STAR framework behavioral impact storytelling',
  ],
};

export const initialResumeSuggestions: ResumeSuggestion[] = [
  {
    id: 'sug-1',
    category: 'impact',
    title: 'Add measurable business impact to TechFlow Labs',
    currentText:
      'Engineered customer-facing RESTful APIs in Python/Django and PostgreSQL supporting 250k daily active users.',
    suggestedText:
      'Engineered high-throughput RESTful APIs in Python/Django and PostgreSQL supporting 250k DAU, maintaining 99.98% uptime and decreasing server error rates by 40%.',
    explanation:
      'Quantifying the reliability impact (99.98% uptime, 40% error reduction) demonstrates operational excellence sought after by Nova Technologies.',
    status: 'pending',
  },
  {
    id: 'sug-2',
    category: 'tech',
    title: 'Highlight Kafka and Event-Driven architecture',
    currentText:
      'Designed asynchronous worker queues with Celery and Redis to handle batch email notifications.',
    suggestedText:
      'Designed asynchronous event queues with Celery, Redis, and event pub/sub streams to handle 5M+ daily batch notifications with zero message loss.',
    explanation:
      'The target job emphasizes distributed streaming and pub/sub patterns. Clarifying message guarantees directly addresses their preferred skills.',
    status: 'pending',
  },
  {
    id: 'sug-3',
    category: 'action_verb',
    title: 'Strengthen action verb for AWS migration',
    currentText:
      'Migrated monolithic data pipeline to containerized Kubernetes microservices on AWS EKS, reducing compute overhead by 28%.',
    suggestedText:
      'Spearheaded the zero-downtime migration of monolithic data pipelines to containerized AWS EKS microservices, slashing annual infrastructure costs by $75k (28%).',
    explanation:
      'Starting with "Spearheaded" and citing monetary cost savings elevates engineering leadership visibility.',
    status: 'pending',
  },
];

export const mockQuestions: InterviewQuestion[] = [
  {
    id: 'q-1',
    index: 1,
    total: 4,
    type: 'behavioral',
    questionText:
      'Welcome Alex! Let’s begin. Tell me about yourself, your recent focus at CloudScale Systems, and why you are specifically interested in joining Nova Technologies as a Software Engineer.',
    followUpQuestion:
      'That’s a great background. When transitioning from monolithic services to distributed microservices at CloudScale, what was the most surprising engineering friction you encountered with your team?',
    clarificationTips: [
      'Focus on your distributed systems background',
      'Mention your enthusiasm for Nova Technologies’ platform scale',
      'Keep response between 90–120 seconds',
    ],
    candidateTranscript:
      'Hi, thanks for having me today. Over the past five years, I’ve specialized in backend systems and high-throughput microservices, most recently as a Senior Backend Engineer at CloudScale Systems. There, I led the architecture of our real-time ingestion pipeline in Python and FastAPI, scaling it to handle over 40 million events per day while dropping p99 latency down to 42 milliseconds. I’ve been following Nova Technologies closely—especially your work solving developer infrastructure orchestration—and I’m excited to bring my experience with asynchronous systems and scalable API design to your platform team.',
    candidateAnswerTime: 94,
    evaluation: {
      score: 92,
      strengths: [
        'Strong, articulate introduction connecting past architectural wins to Nova’s mission',
        'Directly quoted concrete metrics (40M events/day, 42ms p99 latency)',
        'Clear and confident tone matching senior engineering expectations',
      ],
      improvements: [
        'Could briefly mention collaboration with cross-functional product stakeholders',
        'Consider sharing a quick personal insight into what sparked your interest in developer tools',
      ],
      starFeedback: {
        situation: 'Leading ingestion architecture at CloudScale Systems.',
        task: 'Scaling data pipelines under rising traffic demands.',
        action: 'Leveraged FastAPI, async workers, and optimized schema tuning.',
        result: 'Processed 40M+ daily events and reduced p99 latency to 42ms.',
      },
    },
  },
  {
    id: 'q-2',
    index: 2,
    total: 4,
    type: 'technical',
    questionText:
      'Can you tell me about a technically challenging project you worked on where you encountered an unexpected architectural or performance bottleneck, and how you diagnosed and resolved it?',
    followUpQuestion:
      'You mentioned database connection pool exhaustion during sudden traffic spikes. What monitoring signals did you use to differentiate between slow queries and pool starvation, and how did you prevent cascade failures?',
    clarificationTips: [
      'Use the STAR framework (Situation, Task, Action, Result)',
      'Explain your diagnostic tooling (logs, APM, metrics)',
      'Discuss trade-offs between caching vs query optimization',
    ],
    candidateTranscript:
      'At CloudScale, our primary FastAPI service started throwing 504 gateway timeouts during flash traffic surges. Upon inspecting our Datadog APM and PostgreSQL pg_stat_activity, I discovered that our database connection pool was saturating. Individual workers were holding connections open while executing secondary HTTP calls to third-party endpoints. To resolve this, I refactored the pipeline to decouple external calls from transactional boundaries using an asynchronous task queue, and implemented pgBouncer for transactional connection pooling. This eliminated pool starvation and cut CPU utilization by 35%.',
    candidateAnswerTime: 110,
    evaluation: {
      score: 89,
      strengths: [
        'Demonstrated systematic root-cause analysis using APM and database telemetry',
        'Identified the anti-pattern (holding open DB connections during external I/O)',
        'Implemented industry-standard architectural remedy with pgBouncer and async decoupling',
      ],
      improvements: [
        'Could elaborate on failure modes—e.g. circuit breaker fallback behaviors if pgBouncer itself reached limits',
      ],
      starFeedback: {
        situation: 'Gateway timeouts during unexpected traffic surges.',
        task: 'Diagnose whether bottleneck was query latency or connection saturation.',
        action: 'Analyzed pg_stat_activity, decoupled external I/O, deployed pgBouncer.',
        result: 'Completely eliminated 504 timeouts and cut DB CPU load by 35%.',
      },
    },
  },
  {
    id: 'q-3',
    index: 3,
    total: 4,
    type: 'system_design',
    questionText:
      'Suppose Nova Technologies needs a public-facing API gateway that handles 50,000 requests per second across hundreds of customer endpoints. How would you design this service to enforce rate limiting, ensure low latency, and maintain high availability?',
    clarificationTips: [
      'Clarify traffic patterns (bursty vs steady)',
      'Choose an appropriate rate limiting algorithm (Token Bucket / Sliding Window)',
      'Address distributed state storage (Redis clusters) and local caching',
    ],
    candidateTranscript:
      'To handle 50,000 requests per second with low latency, I would design a tiered gateway architecture. At the edge, an Envoy or Nginx reverse proxy terminates TLS and distributes requests across stateless gateway workers. For rate limiting, I would choose a Sliding Window Counter algorithm backed by a Redis Cluster running Lua scripts for atomic increments. To prevent Redis from becoming a bottleneck at 50k RPS, each gateway worker maintains a short in-memory local cache with token-bucket leases. For resilience, if the Redis tier degrades, the gateway fails open with conservative local limits rather than dropping legitimate customer traffic.',
    candidateAnswerTime: 125,
    evaluation: {
      score: 88,
      strengths: [
        'Excellent architectural decomposition (edge termination, stateless workers, Redis cluster)',
        'Addressed critical scale nuance: preventing Redis bottleneck via local memory leases',
        'Included practical resilience strategy (fail-open vs fail-closed considerations)',
      ],
      improvements: [
        'Could discuss how rate limit tiering applies differently to authenticated users vs unauthenticated IP limits',
      ],
    },
  },
  {
    id: 'q-4',
    index: 4,
    total: 4,
    type: 'coding',
    questionText:
      'Now let’s move to a practical coding challenge. In our coding workspace on your right, you have the problem statement: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target." Take your time, explain your approach, and run your tests when ready.',
    clarificationTips: [
      'Ask clarifying questions if needed (can there be duplicate numbers? is there always exactly one solution?)',
      'Aim for O(n) time complexity using a hash map',
      'Test your code with both positive and negative integers',
    ],
    codingTask: {
      problemTitle: 'Two Sum',
      problemDescription:
        'Given an array of integers `nums` and an integer `target`, return the indices of the two numbers such that they add up to `target`.\n\nYou may assume that each input will have exactly one solution, and you may not use the same element twice. You can return the answer in any order.',
      requirements: [
        'Time Complexity: Target O(n) runtime',
        'Space Complexity: Target O(n) auxiliary space',
        'Return zero-indexed indices [index1, index2]',
      ],
      constraints: [
        '2 <= nums.length <= 10^4',
        '-10^9 <= nums[i] <= 10^9',
        '-10^9 <= target <= 10^9',
        'Only one valid answer exists.',
      ],
      examples: [
        {
          input: 'nums = [2, 7, 11, 15], target = 9',
          output: '[0, 1]',
          explanation: 'Because nums[0] + nums[1] == 2 + 7 == 9, we return [0, 1].',
        },
        {
          input: 'nums = [3, 2, 4], target = 6',
          output: '[1, 2]',
          explanation: 'Because nums[1] + nums[2] == 2 + 4 == 6, we return [1, 2].',
        },
        {
          input: 'nums = [3, 3], target = 6',
          output: '[0, 1]',
        },
      ],
      starterCode: {
        python: `def two_sum(nums: list[int], target: int) -> list[int]:
    # Write your solution here
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
`,
        javascript: `function twoSum(nums, target) {
    // Write your solution here
    const seen = new Map();
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (seen.has(complement)) {
            return [seen.get(complement), i];
        }
        seen.set(nums[i], i);
    }
    return [];
}
`,
        typescript: `function twoSum(nums: number[], target: number): number[] {
    // Write your solution here
    const seen = new Map<number, number>();
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (seen.has(complement)) {
            return [seen.get(complement)!, i];
        }
        seen.set(nums[i], i);
    }
    return [];
}
`,
      },
      testCases: [
        {
          id: 1,
          input: 'nums = [2, 7, 11, 15], target = 9',
          expected: '[0, 1]',
        },
        {
          id: 2,
          input: 'nums = [3, 2, 4], target = 6',
          expected: '[1, 2]',
        },
        {
          id: 3,
          input: 'nums = [3, 3], target = 6',
          expected: '[0, 1]',
        },
        {
          id: 4,
          input: 'nums = [-1, -2, -3, -4, -5], target = -8',
          expected: '[2, 4]',
          hidden: true,
        },
      ],
      solutionHint:
        'A single pass hash map allows you to check if the complement (target - current_num) already exists in O(1) average lookup time.',
    },
    submittedCode: `def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`,
    codeLanguage: 'python',
    testCasesPassed: 4,
    totalTestCases: 4,
    candidateAnswerTime: 245,
    evaluation: {
      score: 95,
      strengths: [
        'Optimal O(N) time complexity and O(N) space complexity single-pass hash map',
        'Properly handled negative integers and duplicate values',
        'Clean variable naming (`complement`, `seen`) with clear algorithmic explanation',
      ],
      improvements: [
        'Could include a quick guard check validating minimum input array length (nums.length >= 2)',
      ],
      codingEvaluation: {
        timeComplexity: 'O(N) - Single pass through the array',
        spaceComplexity: 'O(N) - Hash map storing up to N elements',
        cleanliness: 'Production-ready, idiomatic Python dictionary usage',
        edgeCasesHandled: true,
      },
    },
  },
];

export const sampleCompletedReport: InterviewReport = {
  id: 'rep-nova-01',
  role: 'Software Engineer - Platform & Core APIs',
  company: 'Nova Technologies',
  date: 'Sep 21, 2026',
  durationSeconds: 1420, // ~23m 40s
  overallScore: 88,
  verdict: 'Strong Hire',
  categoryScores: {
    jobRelevance: 92,
    technicalKnowledge: 89,
    answerQuality: 88,
    communication: 91,
    problemSolving: 90,
  },
  speechMetrics: {
    wordsPerMinute: 136,
    longPauses: 2,
    averageDurationSeconds: 104,
    starAdherenceScore: 92,
  },
  strongAreas: [
    'Articulate architectural storytelling with concrete metrics (40M events/day, 42ms p99 latency)',
    'Solid distributed systems intuition regarding connection pooling, Redis caching, and async decoupling',
    'Flawless O(N) algorithmic coding execution passing all edge cases and hidden test suites',
    'Calm, structured communication adhering to the STAR method under technical inquiry',
  ],
  areasToImprove: [
    'Deepen coverage of cross-functional stakeholder collaboration when describing technical initiatives',
    'Include explicit circuit-breaking and graceful degradation mechanics in system design proposals',
    'Add input validation sanity checks in coding solutions prior to core loop execution',
  ],
  recommendedPractice: [
    'Practice System Design: Tiered rate limiting with distributed Redis clusters and edge lease tokens',
    'Behavioral: Refine STAR narrative for cross-team conflict resolution scenarios',
    'Algorithms: Sliding window and dynamic programming interview problems',
  ],
  questions: mockQuestions,
};

export const samplePastInterviews = [
  {
    id: 'int-1',
    role: 'Software Engineer - Platform',
    company: 'Nova Technologies',
    date: 'Sep 21, 2026',
    type: 'Full Interview',
    score: 88,
    duration: '24 min',
    verdict: 'Strong Hire',
  },
  {
    id: 'int-2',
    role: 'Backend Infrastructure Engineer',
    company: 'Stripe',
    date: 'Sep 14, 2026',
    type: 'Technical & Coding',
    score: 84,
    duration: '28 min',
    verdict: 'Hire',
  },
  {
    id: 'int-3',
    role: 'Senior Platform Engineer',
    company: 'Vercel',
    date: 'Aug 29, 2026',
    type: 'System Design & Behavioral',
    score: 81,
    duration: '22 min',
    verdict: 'Hire',
  },
];
